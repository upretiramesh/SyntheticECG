import torch
from autoencoder import AutoEncoder
from torch.utils.data import Dataset, DataLoader
import  pandas as pd
import numpy as np
import os
import wfdb
from  glob import glob


modelAE = AutoEncoder()
modelAE.load_state_dict(torch.load('./autoencoder_train_epoch.pt')['ae_state_dict'])
modelAE.eval()
print('model loaded')


class CustomDataset(Dataset):
    """
    This custom class prepared the dataset to be used in the model.

    Argument:
        base_path : location of the directory where required csv file and all required files are located
        file_path : the ptbxl_database.csv file which contains patients information and corresponding ecg file name
        sampling_rate : the dataset contains two sampling rate (100 and 500), by default 500

    Output:
        ECG signals of patient and gender
    """

    def __init__(self, base_path, file_path, sampling_rate=500, max_val=0):
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.df = pd.read_csv(self.base_path + file_path)
        self.df= self.df.query("super_class=='NORM'")
        self.max_val =max_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.sampling_rate == 100:
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -2], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        else:
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -1], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]

        data = torch.tensor(data.T, dtype=torch.float32)/self.max_val
        return data


class LoadFakeData(Dataset):
    def __init__(self, input):
        self.input = input

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        path = self.input[index]

        ecg = pd.read_csv(path, sep=' ', header=None)
        if ecg.shape[0]==12:
            ecg = ecg.loc[(0, 1, 6, 7, 8, 9, 10, 11), :]
        ecg = np.array(ecg.values)
        ecg = ecg/35
        ecg = torch.tensor(ecg, dtype=torch.float32)
        return ecg



def extract_laten(input_data_path=None, out_path=None):

    input_path = f'./samples/' if input_data_path is None else input_data_path # eg. /samples/*
    output_path = f'./latent/' if out_path is None else out_path
    os.makedirs(output_path, exist_ok=True)

    fake = LoadFakeData(glob(input_path))
    fakeloader = DataLoader(fake, batch_size=64)

    data = []
    with torch.inference_mode():
        for i, d in enumerate(fakeloader):
            ecg = modelAE(d, latent=True)
            data.append(ecg)
            
    data = torch.cat(data, dim=0)
    torch.save(data, output_path+'latent.pt')
    print('completed')


def original_data_laten():
    base_dir = '/home/rameshu/D1/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    filename2 = 'preprocess_ptbxl_filename.csv'

    dataset =  CustomDataset(base_path=base_dir, file_path=filename2, max_val=35)

    output_path = f'./latent/'
    os.makedirs(output_path, exist_ok=True)

    fakeloader = DataLoader(dataset, batch_size=64)
    
    data = []
    with torch.inference_mode():

        for i, d in enumerate(fakeloader):
            ecg = modelAE(d, latent=True)
            data.append(ecg)
            
    data = torch.cat(data, dim=0)
    
    torch.save(data, output_path+f'original_data_latent.pt')
    print(f'completed')
