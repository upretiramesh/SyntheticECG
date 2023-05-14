# Reference: https://github.com/vlbthambawita/Pulse2Pulse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def TargetEmbedding(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(0.3), 
        nn.LayerNorm(out_features),
        nn.Linear(out_features, out_features),
        nn.LeakyReLU(0.3), 
        nn.LayerNorm(out_features),
        nn.Linear(out_features, out_features)
    )


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle
    


class AdvP2PDiscriminatorCond(nn.Module):
    def __init__(self, model_size=50, num_channels=8, shift_factor=2, alpha=0.2):
        super(AdvP2PDiscriminatorCond, self).__init__()
        self.model_size = model_size 
        self.num_channels = num_channels  
        self.shift_factor = shift_factor  #
        self.alpha = alpha

        self.conv1 = nn.Conv1d(num_channels+1,  model_size, 25, stride=2, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 5 * model_size, 25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(5 * model_size, 10 * model_size, 25, stride=2, padding=11)
        self.conv5 = nn.Conv1d(10 * model_size, 20 * model_size, 25, stride=4, padding=11)
        self.conv6 = nn.Conv1d(20 * model_size, 25 * model_size, 25, stride=4, padding=11)
        self.conv7 = nn.Conv1d(25 * model_size, 100 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)
        self.ps6 = PhaseShuffle(shift_factor)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(25000, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):

        targets = targets.repeat(1, 5000//5)
        targets = targets.unsqueeze(1)

        x = torch.cat([x, targets], dim=1)

        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        x = self.ps5(x)

        x = F.leaky_relu(self.conv6(x), negative_slope=self.alpha)
        x = self.ps6(x)

        x = F.leaky_relu(self.conv7(x), negative_slope=self.alpha)


        x = self.flat(x)

        return self.fc1(x)
    
class AdvP2PDiscriminatorCondV2(nn.Module):
    def __init__(self, model_size=50, num_channels=8, shift_factor=2, alpha=0.2):
        super(AdvP2PDiscriminatorCondV2, self).__init__()
        self.model_size = model_size  
        self.num_channels = num_channels  
        self.shift_factor = shift_factor  
        self.alpha = alpha

        self.condition_info = nn.Sequential(
                                            nn.Linear(5, 5000), 
                                            nn.LeakyReLU(negative_slope=alpha))

        self.conv1 = nn.Conv1d(num_channels+1,  model_size, 25, stride=2, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 5 * model_size, 25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(5 * model_size, 10 * model_size, 25, stride=2, padding=11)
        self.conv5 = nn.Conv1d(10 * model_size, 20 * model_size, 25, stride=4, padding=11)
        self.conv6 = nn.Conv1d(20 * model_size, 25 * model_size, 25, stride=4, padding=11)
        self.conv7 = nn.Conv1d(25 * model_size, 100 * model_size, 25, stride=4, padding=11)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)
        self.ps6 = PhaseShuffle(shift_factor)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(25000, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):

        targets = self.condition_info(targets)

        targets = targets.unsqueeze(1)

        x = torch.cat([x, targets], dim=1)

        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        x = self.ps5(x)

        x = F.leaky_relu(self.conv6(x), negative_slope=self.alpha)
        x = self.ps6(x)

        x = F.leaky_relu(self.conv7(x), negative_slope=self.alpha)
        
        x = self.flat(x)

        return self.fc1(x)
    

class AdvP2PDiscriminator(nn.Module):
    def __init__(self, model_size=50, num_channels=8, shift_factor=2, alpha=0.2):
        super(AdvP2PDiscriminator, self).__init__()
        self.model_size = model_size 
        self.num_channels = num_channels 
        self.shift_factor = shift_factor  
        self.alpha = alpha

        self.conv1 = nn.Conv1d(num_channels,  model_size, 25, stride=2, padding=11)
        self.conv2 = nn.Conv1d(model_size, 2 * model_size, 25, stride=2, padding=11)
        self.conv3 = nn.Conv1d(2 * model_size, 5 * model_size, 25, stride=2, padding=11)
        self.conv4 = nn.Conv1d(5 * model_size, 10 * model_size, 25, stride=2, padding=11)
        self.conv5 = nn.Conv1d(10 * model_size, 20 * model_size, 25, stride=4, padding=11)
        self.conv6 = nn.Conv1d(20 * model_size, 25 * model_size, 25, stride=4, padding=11)
        self.conv7 = nn.Conv1d(25 * model_size, 100 * model_size, 25, stride=4, padding=11)
        self.conv8 = nn.Conv1d(100 * model_size+1, 200 * model_size, 5)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)
        self.ps6 = PhaseShuffle(shift_factor)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(10000, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        x = self.ps5(x)

        x = F.leaky_relu(self.conv6(x), negative_slope=self.alpha)
        x = self.ps6(x)

        x = F.leaky_relu(self.conv7(x), negative_slope=self.alpha)

        targets = targets.unsqueeze(1)
        x = torch.concat([x, targets], dim=1)

        x = F.leaky_relu(self.conv8(x), negative_slope=self.alpha)

        x = self.flat(x)

        return self.fc1(x)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, st, pd, laten_emb, alpha=0.2):
        super(Block, self).__init__()
        self.con_layer = nn.Conv1d(in_ch,  out_ch, 25, stride=st, padding=pd)
        self.act = nn.LeakyReLU(negative_slope=alpha)

        self.linear_target_emb = nn.Sequential(
            nn.LeakyReLU(negative_slope=alpha),
            nn.Linear(laten_emb, out_ch * 2)
            )

    def forward(self, x, target_emb):
        out = self.act(self.con_layer(x))
        # condition
        target_emb = self.linear_target_emb(target_emb)
        target_emb = rearrange(target_emb, 'b c -> b c 1')
        target_maps = target_emb.chunk(2, dim=1)
        scale, shift = target_maps
        out = out * (scale + 1) + shift

        return self.act(out)


class AdvP2PDiscriminatorCondV3(nn.Module):
    def __init__(self, model_size=50, ngpus=1, num_channels=8, shift_factor=2, alpha=0.2):
        super(AdvP2PDiscriminatorCondV3, self).__init__()
        self.model_size = model_size 
        self.num_channels = num_channels  
        self.shift_factor = shift_factor 
        self.alpha = alpha

        latent_dim = model_size * 2
        self.linear_embedding = TargetEmbedding(5, latent_dim)

        self.conv1 = Block(num_channels,  model_size, st=2, pd=11, laten_emb=latent_dim)
        self.conv2 = Block(model_size, 2 * model_size, st=2, pd=11, laten_emb=latent_dim)
        self.conv3 = Block(2 * model_size, 5 * model_size, st=2, pd=11, laten_emb=latent_dim)
        self.conv4 = Block(5 * model_size, 10 * model_size, st=2, pd=11, laten_emb=latent_dim)
        self.conv5 = Block(10 * model_size, 20 * model_size, st=4, pd=11, laten_emb=latent_dim)
        self.conv6 = Block(20 * model_size, 25 * model_size, st=4, pd=11, laten_emb=latent_dim)
        self.conv7 = Block(25 * model_size, 100 * model_size, st=4, pd=11, laten_emb=latent_dim)

        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.ps5 = PhaseShuffle(shift_factor)
        self.ps6 = PhaseShuffle(shift_factor)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(25000, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):
        targets = self.linear_embedding(targets)

        x = self.conv1(x, targets)
        x = self.ps1(x)

        x = self.conv2(x, targets)
        x = self.ps2(x)

        x = self.conv3(x, targets)
        x = self.ps3(x)

        x = self.conv4(x, targets)
        x = self.ps4(x)

        x = self.conv5(x, targets)
        x = self.ps5(x)

        x = self.conv6(x, targets)
        x = self.ps6(x)

        x = self.conv7(x, targets)


        x = self.flat(x)

        return self.fc1(x)