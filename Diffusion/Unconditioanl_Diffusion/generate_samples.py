import torch
from diffusion import GaussianDiffusion
from diffusion_model import ECGUnetModel
import os
import numpy as np

model = ECGUnetModel(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()

leads = 8  # 12
milestone = 174
nun_samples = 10000
    
diffusion_model = GaussianDiffusion(model, channels = 8, timesteps = 500, loss_type = 'l1').cuda()

# milestone = 149
results_folder = f'./pretrained/'

data = torch.load(results_folder + f'model-{milestone}.pt')
diffusion_model.load_state_dict(data['ema'])
print('model loaded successfully')

path_to_save_array = f'./gen_samples/chk_{milestone}/'
os.makedirs(path_to_save_array, exist_ok=True)

sample=0
batch_size = 25
epochs = nun_samples//25

for i in range(epochs):
    fake_to_plot = diffusion_model.sample(batch_size=batch_size)
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
        # append for future
        np.savetxt(path_to_save_array+f'{sample}.asc', ecg)
        sample +=1
    print(f'generated sample batch of {i} for model timesteps:{ts}')
print(f'Timesteps of {ts} is generated')