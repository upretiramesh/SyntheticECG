import torch
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import wfdb
from torchvision import models
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

np.random.seed(123)
torch.manual_seed(1234)

data = []
model = 'AdvP2P'
ch = 2500



# load real ecg
base_dir = '/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
filename2 = 'preprocess_ptbxl_filename.csv'
df_real = pd.read_csv(base_dir + filename2)
df_real['target'] = 1
df_real.filename_hr = df_real.filename_hr.apply(lambda x: base_dir+x)
df_real = df_real[['filename_hr', 'target']]
real = df_real.sample(10025)

# load fake ECG
checkpoint = f'/samples/*'
fake = pd.DataFrame({'filename_hr':glob(checkpoint), 'target':0})

# combine data
df = pd.concat([real, fake])
df = shuffle(df)
df = shuffle(df)
X = df['filename_hr']
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

# Data loader
class LoadData(Dataset):
    def __init__(self, input, output, real=True):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        path = self.input.iloc[index]
        t = self.output.iloc[index]

        if not real:
            ecg = pd.read_csv(path, sep=' ', header=None)
            if ecg.shape[0]==12:
                ecg = ecg.loc[(0, 1, 6, 7, 8, 9, 10, 11), :]
            ecg = np.abs(ecg.values)
        else:
            ecg = wfdb.rdsamp(path, channel_names=['I','II','V1','V2','V3','V4','V5','V6'])[0]
            ecg = np.abs(ecg)
            ecg = ecg.T

        ecg = ecg/32.8
        ecg = torch.tensor(ecg, dtype=torch.float32)
        ecg = ecg.flatten().view(200, 200)
        ecg = ecg.expand(3, 200, 200)

        t = torch.tensor(t).type(torch.LongTensor)
        return ecg, t


train_data = LoadData(X_train, y_train)
test_data = LoadData(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


feature_extract = True
use_pretrained = True
num_classes = 2

model_ft = models.resnet18(weights='ResNet18_Weights.DEFAULT') 
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
input_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)

# train model
for epoch in range(10):
    epoch_loss = 0
    for i, (ecgs, labels) in enumerate(train_loader):
        ecgs = ecgs.to(device)
        labels = labels.to(device)
        ecgs = torch.nn.functional.interpolate(ecgs, size=224)
        optimizer.zero_grad()
        outputs = model_ft(ecgs)
        loss = criterion(outputs, labels)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    print(f'\n## {model}-{ch}- Epoch {epoch} loss: {epoch_loss} ##\n')

print('Training completed')

# test result
with torch.inference_mode():
    accuracy = 0
    f1score = 0
    auc = 0
    pscore = 0
    rscore = 0
    for i, (ecgs, labels) in enumerate(test_loader):
        ecgs = ecgs.to(device)
        labels = labels.to(device)
        ecgs = torch.nn.functional.interpolate(ecgs, size=224)
        outputs = model_ft(ecgs)
        _, outputs = torch.max(outputs.data, 1)
        outputs = outputs.detach().cpu().numpy()
        labels= labels.detach().cpu().numpy()
        acc = accuracy_score(outputs, labels)
        f1s = f1_score(outputs, labels)
        # rocauc = roc_auc_score(outputs, labels)
        ps = precision_score(outputs, labels)
        rs = recall_score(outputs, labels)

        accuracy += acc
        f1score += f1s
        # auc += rocauc
        pscore += ps
        rscore += rs
    print('Final Result')
    print('Accuracy: ', accuracy/len(test_loader))
    print('F1 score: ', f1score/len(test_loader))
    # print('ROC AUC score: ', auc/len(test_loader))
    print('Precision Score: ', pscore/len(test_loader))
    print('Recall Score: ', rscore/len(test_loader))

    data.append({'model':model, 'checkpoint':ch, 'acc':accuracy/len(test_loader), 'f1s':f1score/len(test_loader), 'ps':pscore/len(test_loader), 'rs':rscore/len(test_loader)})

pd.DataFrame(data).to_csv(f'{model}_classifier_result.csv', index=False)




