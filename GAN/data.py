import torch
import torch.utils.data as data
import wfdb
import pandas as pd
import numpy as np


class CustomDataset(data.Dataset):
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
        # target = self.df.iloc[idx, 3]
        if self.sampling_rate == 100:
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -2], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        else:
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, -1], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]

        data = torch.tensor(data.T, dtype=torch.float32)/self.max_val if self.max_val>0 else torch.tensor(data.T, dtype=torch.float32)
        return data 


class CustomDatasetCond(data.Dataset):
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
        self.df= self.df.query("super_class=='NORM'")

        self.target = pd.read_csv(target_path, sep=';')
        self.target = self.target.iloc[self.df.ecg_id.to_numpy()-1]

        self.target['P_wave'] = 2 * (self.target['P_Offset']-self.target['P_Onset'])
        features = ['QRS_Duration', 'QT_Interval', 'PR_Interval', 'AvgRRInterval','P_wave']
        self.target = self.target[features]
        self.max_val = max_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        target = torch.tensor(self.target.iloc[idx], dtype=torch.float32)/500

        if self.sampling_rate == 100:
            cid = self.df.columns.get_loc("filename_lr")
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, cid], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
        else:
            cid = self.df.columns.get_loc("filename_hr")
            data = wfdb.rdsamp(self.base_path + self.df.iloc[idx, cid], channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]

        data = torch.tensor(data.T, dtype=torch.float32) / self.max_val if self.max_val > 0 else torch.tensor(data.T,
                                                                                                              dtype=torch.float32)
        return data, target

