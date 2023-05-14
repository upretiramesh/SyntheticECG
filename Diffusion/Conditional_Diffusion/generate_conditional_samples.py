import torch
from diffusion_model_condition import ECGUnetCondition
from diffusion_conditional import GaussianDiffusion
import os
import numpy as np
from torch.utils.data import DataLoader
from utils import LoadPTBXLDatasetMUSE, LoadPTBXLDatasetClass


condition = 'muse'
model = ECGUnetCondition(dim = 64, dim_mults = (1, 2, 4, 8), condition=condition).cuda()
diffusion_model = GaussianDiffusion(model, channels = 8, timesteps = 500,  loss_type = 'l1').cuda()

if condition == 'muse':
    milestone = 124
    results_folder = f'./pretrained/muse_features/'
else:
    milestone = 174
    results_folder = f'./pretrained/class_labels/'

data = torch.load(results_folder + f'model-{milestone}.pt')
diffusion_model.load_state_dict(data['ema'])
print(f'model loaded successfully')

path_to_save_array = f'./gen_samples/{condition}/chk_{milestone}/'
os.makedirs(path_to_save_array, exist_ok=True)


def cycle(dl):
    while True:
        for data in dl:
            yield data


base_dir = '/work/ramesh/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
filename = 'preprocess_ptbxl_filename.csv'
targetpath = '/work/ramesh/final_experiment/cddm/PTB-XL-MUSE-Analysis/MUSE/tstECGMeasMatrix.csv'

bs = 25
if condition == 'muse':
    ds = LoadPTBXLDatasetMUSE(base_path=base_dir, file_path=filename, target_path=targetpath, max_val=32.8)
else:
    ds = LoadPTBXLDatasetClass(base_path=base_dir, file_path=filename, max_val=32.8)

dl = cycle(DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=True))

sample=0
nun_samples = 10000
epochs = nun_samples//25
leads = 8

for i in range(epochs):
    _, ground_truth = next(dl)
    fake_to_plot = diffusion_model.sample(batch_size=bs, class_labels=ground_truth)
    fake_ecg = fake_to_plot.cpu().numpy()
    for idx in range(fake_to_plot.shape[0]):
        ecg = fake_ecg[idx]

        if leads == 12:
            # create additional 4 leads
            III = np.reshape((ecg[1] - ecg[0]), (1, -1))
            aVR = np.reshape((-0.5*(ecg[1] + ecg[0])), (1, -1))
            aVL = np.reshape((ecg[0] - 0.5 * ecg[1]), (1, -1))
            aVF = np.reshape((ecg[1] - 0.5 * ecg[0]), (1, -1))

            # combine all leads into single
            ecg = np.concatenate([ecg[:2], III, aVR, aVL, aVF, ecg[2:]])

        # Inverser back to original form
        ecg = ecg*32.8
        np.savetxt(path_to_save_array+f'{sample}.asc', ecg)
        sample +=1
print(f'generated sample for checkpoint {milestone}')