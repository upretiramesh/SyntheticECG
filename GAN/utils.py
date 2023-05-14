import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
matplotlib.use('agg')


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda, use_cuda=False):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else
                              torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_gradient_penalty_cond(net_dis, real_data, fake_data, batch_size, lmbda, targets_info, use_cuda=False):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates, targets_info)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else
                              torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def plot_single_ecg(true_ecg, fake_ecg):
    fig, axs = plt.subplots( 2)
    true_ecg_8_chs = true_ecg.reshape(8, 5000)
    fake_ecg_8_chs = fake_ecg.reshape(8, 5000)
    
    for i in range(8):
        sns_plot = sns.lineplot(x = [i for i in range(len(true_ecg_8_chs[i]))], y=true_ecg_8_chs[i], ax = axs[0])
        sns_plot = sns.lineplot(x = [i for i in range(len(fake_ecg_8_chs[i]))], y=fake_ecg_8_chs[i], ax = axs[1])
        
    fig = sns_plot.get_figure()
    fig.set_size_inches(11.7, 15)
    return fig
    

def plot_multiple_ecg(true_ecgs, fake_ecgs, num_of_plots=4):

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


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len=8, embed_model_dim=5000):
        """
        Args:
            max_seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x


class AutoPositionEmbedding(nn.Module):

    def __init__(self, num_embeddings=8, embedding_dim=5000):
        super(AutoPositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        seq_len = x.shape[1]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        return x + embeddings