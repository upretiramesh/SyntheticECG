from glob import glob
import numpy as np
import torch
import os


def get_psa():
    input_path = f'samples/AdvP2P/ch_2500/*'
    output_path = f'./fft/'
    os.makedirs(output_path, exist_ok=True)
    
    data = []
    
    for fname in glob(input_path):
        ecg = np.loadtxt(fname)
        ecg = torch.from_numpy(ecg)
        fft = torch.abs(torch.fft.rfft2(ecg))**2
        data.append(fft)
    data = torch.cat(data, dim=0)
    torch.save(data, output_path+'psa_advp2p.pt')
    print(f'completed')
      
get_psa()