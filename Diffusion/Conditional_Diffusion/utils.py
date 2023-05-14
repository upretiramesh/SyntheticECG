
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
import pandas as pd
import wfdb
import torch



def get_plots_all_RHTM_10s(true_ecgs, fake_ecgs, num_of_plots = 4):
    bs = true_ecgs.shape[0]
    if bs > num_of_plots:
        bs = num_of_plots

    fig, axs = plt.subplots(bs, 2)

    for b in range(bs):

        true_ecg_8_chs = true_ecgs[b].reshape(8, 5000)
        fake_ecg_8_chs = fake_ecgs[b].reshape(8, 5000)
    
        for i in range(8):
            sns_plot = sns.lineplot(x = [i for i in range(len(true_ecg_8_chs[i]))], y=true_ecg_8_chs[i], ax = axs[b,0])
            sns_plot = sns.lineplot(x = [i for i in range(len(fake_ecg_8_chs[i]))], y=fake_ecg_8_chs[i], ax = axs[b,1])
            
    fig = sns_plot.get_figure()
    fig.set_size_inches(11.7, 15)
    return fig


def get_plots_RHTM_10s(true_ecg, fake_ecg):
    fig, axs = plt.subplots( 2)
    true_ecg_8_chs = true_ecg.reshape(8, 5000)
    fake_ecg_8_chs = fake_ecg.reshape(8, 5000)
    
    for i in range(8):
        sns_plot = sns.lineplot(x = [i for i in range(len(true_ecg_8_chs[i]))], y=true_ecg_8_chs[i], ax = axs[0])
        sns_plot = sns.lineplot(x = [i for i in range(len(fake_ecg_8_chs[i]))], y=fake_ecg_8_chs[i], ax = axs[1])
        
    fig = sns_plot.get_figure()
    fig.set_size_inches(11.7, 15)
    return fig


class LoadPTBXLDataset(Dataset):
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
        self.max_val =max_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.sampling_rate == 100:
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -2], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        else:
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -1], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]

        data = torch.tensor(data.T, dtype=torch.float32)/self.max_val if self.max_val>0 else torch.tensor(data.T, dtype=torch.float32)
        return data 


class LoadPTBXLDatasetClass(Dataset):
    """
    This custom class prepared the dataset to be used in the model.

    Argument:
        base_path : location of the directory where required csv file and all required files are located
        file_path : the ptbxl_database.csv file which contains patients information and corresponding ecg file name
        sampling_rate : the dataset contains two sampling rate (100 and 500), by default 500

    Output:
        ECG signals of patient and gender
    """

    def __init__(self, base_path, file_path, sampling_rate=500, max_val=0, superclass = ['NORM', 'MI', 'STTC']):
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.df = pd.read_csv(self.base_path + file_path)
        self.superclass = superclass
        self.df = self.df[self.df['super_class'].isin(self.superclass)]
        self.map_superclass = {key:i for i, key in enumerate(self.superclass)}
        self.df['super_class'] = self.df['super_class'].map(self.map_superclass)
        self.max_val = max_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cid = self.df.columns.get_loc("super_class")
        target = self.df.iloc[idx, cid]

        if self.sampling_rate == 100:
            cid = self.df.columns.get_loc("filename_lr")
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, cid], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        else:
            cid = self.df.columns.get_loc("filename_hr")
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, cid], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]

        data = torch.tensor(data.T, dtype=torch.float32) / self.max_val if self.max_val > 0 else torch.tensor(data.T,
                                                                                                              dtype=torch.float32)
        return data, torch.tensor(target).long()
    

class LoadPTBXLDatasetMUSE(Dataset):
    """
    This custom class prepared the dataset to be used in the model.

    Argument:
        base_path : location of the directory where required csv file and all required files are located
        file_path : the ptbxl_database.csv file which contains patients information and corresponding ecg file name
        sampling_rate : the dataset contains two sampling rate (100 and 500), by default 500

    Output:
        ECG signals of patient and gender
    """

    def __init__(self, base_path, file_path, target_path, sampling_rate=500, max_val=0):
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.df = pd.read_csv(self.base_path + file_path)
        self.target = pd.read_csv(target_path, sep=';')
        self.target['P_wave'] = 2 * (self.target['P_Offset']-self.target['P_Onset'])
        features = ['QRS_Duration', 'QT_Interval', 'PR_Interval', 'AvgRRInterval','P_wave']
        self.target = self.target[features]
        self.max_val = max_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        target = torch.tensor(self.target.iloc[idx], dtype=torch.float32)

        if self.sampling_rate == 100:
            cid = self.df.columns.get_loc("filename_lr")
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, cid], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        else:
            cid = self.df.columns.get_loc("filename_hr")
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, cid], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]

        data = torch.tensor(data.T, dtype=torch.float32) / self.max_val if self.max_val > 0 else torch.tensor(data.T,
                                                                                                              dtype=torch.float32)
        return data, target