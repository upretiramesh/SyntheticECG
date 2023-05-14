#===========================================
# Reference: https://github.com/vlbthambawita/Pulse2Pulse
#===========================================

import argparse
import os
from tqdm import tqdm
import numpy as np

#Pytorch
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# from data.ecg_data_loader import ECGDataSimple as ecg_data
from discriminator import AdvP2PDiscriminatorCond
from utils import calc_gradient_penalty_cond, plot_single_ecg, plot_multiple_ecg
from data import CustomDatasetCond
from conditional_gans_generator import AdvP2PSD
from conditional_gans_generator import AdvP2P
from conditional_gans_generator import AdvP2PAutoEmb
from conditional_gans_generator import AdvP2PPosEmb
from conditional_gans_generator import AdvP2PNtFAutoEmb


torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

# define arguments
parser.add_argument("model", type=str, help="Select model name", choices=["AdvP2PSD", "AdvP2P", "AdvP2PAutoEmb", "AdvP2PPosEmb", "AdvP2PNtFAutoEmb"])
parser.add_argument("--exp_name", type=str, required=True, help="A name to the experiment which is used to save checkpoitns and tensorboard output")
parser.add_argument("--out_dir", default="./output", help="Main output dierectory")

parser.add_argument("--tensorboard_dir",  default="./tensorboard", help="Folder to save output of tensorboard")

parser.add_argument("--bs", type=int, default=1, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num_epochs", type=int, default=4000, help="number of epochs of training")

parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("--checkpoint_interval", type=int, default=25, help="Interval to save checkpoint models")

# Checkpoint path to retrain or test models
parser.add_argument("--checkpoint_path", default="", help="Check point path to retrain or test models")
parser.add_argument('--lmbda', default=10.0, help="Gradient penalty regularization factor")

# Action handling 
parser.add_argument("action", type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])
opt = parser.parse_args()


# select training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)

# make subfolder in the output folder 
checkpoint_dir = os.path.join(opt.out_dir, opt.exp_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, opt.exp_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)

# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)


# Prepare Data
def prepare_data():
    # dataset =  ecg_data(opt.data_dirs, norm_num=6000, cropping=None, transform=None)
    # print("Dataset size=", len(dataset))
    base_dir = '/home/rameshu/D1/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    filename2 = 'preprocess_ptbxl_filename.csv'
    targetpath = '/home/rameshu/D1/code/grouth_truth/MUSE/tstECGMeasMatrix.csv'
    dataset =  CustomDatasetCond(base_path=base_dir, file_path=filename2, target_path=targetpath, max_val=35)
    
    dataloader = torch.utils.data.DataLoader( dataset,
        batch_size=opt.bs,
        shuffle=True,
        num_workers=2
    )

    return dataloader

# Save models
def save_model(netG, netD, optimizerG, optimizerD,  epoch):
    py_file_name = os.path.basename(__file__)[:-3]
    check_point_name = py_file_name + "_epoch:{}.pt".format(epoch)
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "netG_state_dict": netG.state_dict(),
        "netD_state_dict": netD.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict()
    }, check_point_path)


# Prepare models
def prepare_model():
    if opt.model=='AdvP2PSD':
        netG = AdvP2PSD()
    elif opt.model=='AdvP2P':
        netG = AdvP2P()
    elif opt.model=='AdvP2PAutoEmb':
        netG = AdvP2PAutoEmb()
    elif opt.model=='AdvP2PPosEmb':
        netG = AdvP2PPosEmb()
    elif opt.model=='AdvP2PNtFAutoEmb':
        netG = AdvP2PNtFAutoEmb()

    netD = AdvP2PDiscriminatorCond()

    netG = netG.to(device)
    netD = netD.to(device)

    return netG, netD

# function to retian from saved checkpoints
def run_train():
    netG, netD = prepare_model()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    dataloaders = prepare_data() 
    train(netG, netD, optimizerG, optimizerD, dataloaders)
    

def run_retrain():

    netG, netD = prepare_model()
    # loading checkpoing
    chkpnt = torch.load(opt.checkpoint_path, map_location="cpu")
    netG.load_state_dict(chkpnt["netG_state_dict"])
    netD.load_state_dict(chkpnt["netD_state_dict"])

    netG = netG.to(device)
    netD = netD.to(device)
    # setup start epoch to checkpoint epoch
    opt.__setattr__("start_epoch", chkpnt["epoch"])

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloaders = prepare_data()
    train(netG, netD, optimizerG, optimizerD, dataloaders)


def train(netG, netD, optimizerG, optimizerD, dataloader):

    for epoch in tqdm(range(opt.start_epoch + 1, opt.num_epochs + 1 - opt.start_epoch)):

        train_G_flag = False
        D_cost_train_epoch = []
        D_wass_train_epoch = []
        G_cost_epoch = []
    
        for i, (sample, targets) in tqdm(enumerate(dataloader, 0)):

            if (i+1) % 2 == 0:
                train_G_flag = True

            # Set Discriminator parameters to require gradients.
            for p in netD.parameters():
                p.requires_grad = True

            one = torch.tensor(1, dtype=torch.float)
            neg_one = one * -1
            one = one.to(device)
            neg_one = neg_one.to(device)

            #############################
            # (1) Train Discriminator
            #############################

            real_ecgs = sample.to(device)
            target_conds = targets.to(device)
            b_size = real_ecgs.size(0)

            netD.zero_grad()

            # Noise
            noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
            noise_Var = noise.to(device)

            D_real = netD(real_ecgs, target_conds)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            fake = netG(noise_Var, target_conds)
            D_fake = netD(fake, target_conds)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty_cond(netD, real_ecgs,
                                                    fake.data, b_size, opt.lmbda, target_conds, use_cuda=True)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            D_cost_train_cpu = D_cost_train.data.cpu()
            D_wass_train_cpu = D_wass_train.data.cpu()

            D_cost_train_epoch.append(D_cost_train_cpu)
            D_wass_train_epoch.append(D_wass_train_cpu)


            #############################
            # (3) Train Generator
            #############################
            if train_G_flag:
                # Prevent discriminator update.
                for p in netD.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                netG.zero_grad()

                # Noise
                noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
                noise_Var = noise.to(device)

                fake = netG(noise_Var, target_conds)
                G = netD(fake, target_conds)
                G = G.mean()

                # Update gradients.
                G.backward(neg_one)
                G_cost = -G

                optimizerG.step()

                G_cost_cpu = G_cost.data.cpu()
                G_cost_epoch.append(G_cost_cpu)
      
                train_G_flag =False

            if i == 0: # take the first batch to plot
                real_ecgs_to_plot = real_ecgs
                fake_to_plot = fake

        D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
        D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
        G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

        
        writer.add_scalar("D_cost_train_epoch_avg",D_cost_train_epoch_avg ,epoch)
        writer.add_scalar("D_wass_train_epoch_avg",D_wass_train_epoch_avg ,epoch)
        writer.add_scalar("G_cost_epoch_avg ",G_cost_epoch_avg  ,epoch)

        print("Epochs:{}\t\tD_cost:{}\t\t D_wass:{}\t\tG_cost:{}".format(
                    epoch, D_cost_train_epoch_avg, D_wass_train_epoch_avg, G_cost_epoch_avg))

         # Save model
        if epoch % opt.checkpoint_interval == 0:
            save_model(netG, netD, optimizerG, optimizerD, epoch)
            fig = plot_single_ecg(real_ecgs_to_plot[0].detach().cpu(), fake_to_plot[0].detach().cpu())
            fig_2 = plot_multiple_ecg(real_ecgs_to_plot.detach().cpu(), fake_to_plot.detach().cpu())

            writer.add_figure("sample", fig, epoch)
            writer.add_figure("sample_batch", fig_2, epoch)
        #fig.savefig("{}.png".format(epoch))


if __name__ == "__main__":

    print(vars(opt))
    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_retrain()
    elif opt.action == "inference":
        print("Inference process is strted..!")
    # Finish tensorboard writer
    writer.close()
