from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import numpy as np
import wfdb


class LoadRealData(Dataset):

    def __init__(self, base_path, file_path, sampling_rate=500, max_val=32.8):
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.df = pd.read_csv(self.base_path + file_path)
        self.df = self.df.iloc[:10000, ]
        self.max_val =max_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -1], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        data = np.abs(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.t()
        data = data/self.max_val
        data = data.flatten().view(200, 200)
        return data.expand(3, 200, 200)


class LoadFakeData(Dataset):
    def __init__(self, path):
        self.df = glob(path+'*')
        self.df = self.df[:10000]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_fine_name = self.df[idx]
        ecg = pd.read_csv(sample_fine_name, sep=' ', header=None)
        if ecg.shape[0]==12:
            ecg = ecg.loc[(0, 1, 6, 7, 8, 9, 10, 11), :]
        ecg = np.abs(ecg.values)
        ecg = ecg/32.8
        ecg = torch.tensor(ecg, dtype=torch.float32)
        ecg = ecg.flatten().view(200, 200)
        return ecg.expand(3, 200, 200)



def get_fid_and_is_score(m, ch, path):
    
    results = []

    # load fake data
    fakedata = LoadFakeData(path)
    fakedataloader = DataLoader(fakedata, batch_size=100, shuffle=False)

    # load real data
    base_dir = '/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    filename2 = 'preprocess_ptbxl_filename.csv'
    realdata = LoadRealData(base_dir, filename2)
    realdataloader = DataLoader(realdata, batch_size=100, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionScore(feature=2048, splits=10, normalize=True).to(device)
    fid = FrechetInceptionDistance(feature=2048, splits=10, normalize=True).to(device)

    for i, (real_ecg, fake_ecg) in enumerate(zip(realdataloader, fakedataloader)):
        # inception for real ecg
        # inception.update(real_ecg)
        # r1, r2 = inception.compute()
        real_ecg = real_ecg.to(device) 
        fake_ecg = fake_ecg.to(device)

        # inception for fake ecg
        inception.update(fake_ecg)
        f1, f2 = inception.compute()

        # fid 
        fid.update(real_ecg, real=True)
        fid.update(fake_ecg, real=False)
        score = fid.compute()
        # fid.reset()

        if i%50==0:
            print({'model':m, 'chk':ch, 'fake_is_mean':f1.item(), 'fake_is_std':f2.item(), 'fid':score.item()})
            # print({'model':m, 'chk':ch, 'real_is_mean':r1.item(), 'real_is_std':r2.item(), 'fake_is_mean':f1.item(), 'fake_is_std':f2.item(), 'fid':score.item()})

        results.append({'model':m, 'chk':ch, 'fake_is_mean':f1.cpu().item(), 'fake_is_std':f2.cpu().item(), 'fid':score.cpu().item()})
    return results


def main():
    model = 'AdvP2P'
    ch = 2500
    checkpoint = f'/samples/{model}/ch_{ch}/'  # pass location of fake data
    result = get_fid_and_is_score(model, ch, checkpoint)
    pd.DataFrame(result).to_csv(f'/work/ramesh/final_experiment/cddm/gen_samples/results/abs_{model}_{ch}.csv', index=False) 
    print(f'process completed for {model} and checkpoint {ch}')

            
if __name__=='__main__':
    main()