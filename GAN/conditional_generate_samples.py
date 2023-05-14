import torch
import os
import numpy as np
from data import CustomDatasetCond
from conditional_gans_generator import AdvP2PSD
from conditional_gans_generator import AdvP2P
from conditional_gans_generator import AdvP2PAutoEmb
from conditional_gans_generator import AdvP2PPosEmb
from conditional_gans_generator import AdvP2PNtFAutoEmb

base_dir = '/home/rameshu/D1/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
filename2 = 'preprocess_ptbxl_filename.csv'
targetpath = '/home/rameshu/D1/code/grouth_truth/MUSE/tstECGMeasMatrix.csv'
dataset =  CustomDatasetCond(base_path=base_dir, file_path=filename2, target_path=targetpath, max_val=35)

dataloader = torch.utils.data.DataLoader( dataset,
    batch_size=25,
    shuffle=True,
    num_workers=2
)


model = 'AdvP2P'
checkpoint = 2500

netG = AdvP2P()

checkpoint_path = f'./output/{model}/checkpoints/conditional_train:{checkpoint}.pt'
path_to_save_ecgs = f'./samples/conditional/{model}/ch_{checkpoint}/'
os.makedirs(path_to_save_ecgs, exist_ok=True)
# loading checkpoing
try: 
    chkpnt = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.load_state_dict(chkpnt["netG_state_dict"])
    print(f'{model} loaded')
    netG = netG.to(device)

    b_size = 25
    sample = 0
        
    netG.eval()
    num = int((10000-sample)/b_size)+1
    
    with torch.inference_mode():
        for i, (_, targets) in enumerate(dataloader):
            noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
            noise_Var = noise.to(device)
            targets = targets.to(device)
            fake_ecg = netG(noise_Var, targets)
            fake_ecg = fake_ecg.cpu().numpy()

            for idx in range(fake_ecg.shape[0]):
                ecg = fake_ecg[idx]
                ecg = ecg * 35
                np.savetxt(path_to_save_ecgs+f'{sample}.asc', ecg)
                sample +=1
    
except Exception as e:
    print(f'Unable to load {model} and checkpoint {checkpoint}\n Error: {e}\n\n')