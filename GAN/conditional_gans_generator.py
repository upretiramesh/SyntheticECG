import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import AutoPositionEmbedding, PositionalEmbedding
from einops import rearrange


class Upsample1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None,):
        super(Upsample1DLayer, self).__init__()
        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same')
    def forward(self, x):
        return self.conv1d(self.upsample_layer(x))


class Upsample1DLayerMulti(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Upsample1DLayerMulti, self).__init__()
        self.upsample_layer = nn.Upsample(scale_factor=upsample)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same')

    def forward(self, x, in_feature):
        x = torch.cat((x, in_feature), 1)
        return self.conv1d((self.upsample_layer(x)))
   

class BICNN(torch.nn.Module):
    def __init__(self, dims, di, kr=25, bi=True):
        super(BICNN, self).__init__()
        self.bi = bi
        self.left_pad = torch.nn.ConstantPad1d(((kr-1)*di, 0), value=0)
        self.cnn1 = torch.nn.Conv1d(dims, dims, kernel_size=kr, dilation=di)
        if self.bi:
            self.right_pad = torch.nn.ConstantPad1d((0, (kr-1)*di), value=0)
            self.cnn2 = torch.nn.Conv1d(dims, dims, kernel_size=kr, dilation=di)
    def forward(self,x):
        left_padded = self.left_pad(x)
        out = self.cnn1(left_padded)
        if self.bi:
            right_padded = self.right_pad(x)
            out_right = self.cnn2(right_padded)
            out = torch.mean(torch.stack([out, out_right]), dim=0)
        return out


class TCN(nn.Module):
    """
    This is a dilated 1d cnn layer stacks with residual connection
    """

    def __init__(self, dimension, dilation_rate=5, kernel=25, down=True, BI=True):
        """
        :type dilation_rate: int --> dilation rate you want to define for program
        :type dimension: int -->  in_channels and out_channel have the same dimension
        :type kernel: int --> size of kernel
        """
        super(TCN, self).__init__()
        self.kernel = kernel
        self.dim = dimension
        layers = [BICNN(self.dim, pow(2,dia), bi=BI) for dia in range(dilation_rate)]
        self.dilation = nn.Sequential(*layers)
        self.act = nn.LeakyReLU() if down else nn.ReLU()

    def forward(self, x):
        out = self.dilation(x)
        out = self.act(out)
        out_x_residual = torch.add(x, out)
        return out_x_residual


class Block(nn.Module):
    def __init__(self, dims, dims_out, down=True, laten_emb=None, bicnn=True):
        super(Block, self).__init__()
        self.cnn1 = nn.Conv1d(dims, dims_out, kernel_size=25, padding='same')
        self.norm = nn.GroupNorm(4, dims_out)

        self.linear_target_emb = nn.Sequential(
            nn.LeakyReLU(0.3),
            nn.Linear(laten_emb, dims_out * 2)
            )

        self.attn = TCN(dims_out, down=down, BI=bicnn)
        self.act = nn.LeakyReLU() if down else nn.ReLU()

    def forward(self, x, target_emb):
        out = self.cnn1(x)
        out = self.norm(out)

        # condition
        target_emb = self.linear_target_emb(target_emb)
        target_emb = rearrange(target_emb, 'b c -> b c 1')
        target_maps = target_emb.chunk(2, dim=1)
        scale, shift = target_maps
        out = out * (scale + 1) + shift

        
        out = self.act(out)
        out = self.attn(out)
        final = self.act(out)
        return final


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


class AdvP2PSD(nn.Module):
    def __init__(self, model_size=64, num_channels=8, upsample=True):
        super(AdvP2PSD, self).__init__()
        self.model_size = model_size  
        self.num_channels = num_channels 

        latent_dim = model_size * 2
        self.linear_embedding = TargetEmbedding(5, latent_dim)
        
        stride = 1
        upsample = 5

        self.up1 = Block(model_size * 5, model_size * 5, down=False, laten_emb=latent_dim, bicnn=False)
        self.up2 = Block(model_size * 3, model_size * 3, down=False, laten_emb=latent_dim, bicnn=False)
        self.up3 = Block(model_size, model_size, down=False, laten_emb=latent_dim, bicnn=False)
        self.up5 = Block(model_size // 2, model_size // 2, down=False, laten_emb=latent_dim, bicnn=False)
        self.up6 = Block(model_size // 4, model_size // 4, down=False, laten_emb=latent_dim, bicnn=False)
        self.up7 = Block(num_channels, num_channels, down=False, laten_emb=latent_dim, bicnn=False)



        self.deconv_1 = Upsample1DLayer(5 * model_size , 5 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Upsample1DLayerMulti(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Upsample1DLayerMulti(3* model_size * 2,  model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Upsample1DLayerMulti(model_size * 2, model_size // 2, 25, stride, upsample=2)
        self.deconv_6 = Upsample1DLayerMulti(model_size // 2 * 2 , model_size // 4, 25, stride, upsample=2)
        self.deconv_7 = Upsample1DLayerMulti(model_size // 4 * 2, num_channels, 25, stride, upsample=2)

        self.last = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=25, padding='same'))

        # new convolutional layers
        self.down1 = Block(num_channels, model_size // 4, laten_emb=latent_dim, bicnn=False)
        self.down2 = Block(model_size // 4, model_size // 2, laten_emb=latent_dim, bicnn=False)
        self.down3 = Block(model_size // 2, model_size, laten_emb=latent_dim, bicnn=False)
        self.down4 = Block(model_size, model_size * 3, laten_emb=latent_dim, bicnn=False)
        self.down5 = Block(model_size * 3, model_size * 5, laten_emb=latent_dim, bicnn=False)
        self.down6 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim, bicnn=False)

        self.conv_1 = nn.Conv1d(model_size // 4, model_size // 4, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 2, model_size // 2, 25, stride=2, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(model_size, model_size , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(model_size * 3, model_size * 3 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)


        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):
        
        encoded_targets = self.linear_embedding(targets)
       
        conv_1_out = self.down1(x, encoded_targets)
        conv_1_out = F.leaky_relu(self.conv_1(conv_1_out))
       
        conv_2_out = self.down2(conv_1_out, encoded_targets)
        conv_2_out = F.leaky_relu(self.conv_2(conv_2_out))
       
        conv_3_out = self.down3(conv_2_out, encoded_targets)
        conv_3_out = F.leaky_relu(self.conv_3(conv_3_out))
        
        conv_4_out = self.down4(conv_3_out, encoded_targets)
        conv_4_out = F.leaky_relu(self.conv_4(conv_4_out))

        conv_5_out = self.down5(conv_4_out, encoded_targets)
        conv_5_out = F.leaky_relu(self.conv_5(conv_5_out))

        conv_6_out = self.down6(conv_5_out, encoded_targets)
        x = F.leaky_relu(self.conv_6(conv_6_out))
      
        x = F.relu(self.deconv_1(x))
        x = self.up1(x, encoded_targets)

        x = F.relu(self.deconv_2(x, conv_5_out))
        x = self.up2(x, encoded_targets)
       
        x = F.relu(self.deconv_3(x, conv_4_out))
        x = self.up3(x, encoded_targets)
        
        x = F.relu(self.deconv_5(x, conv_3_out))
        x = self.up5(x, encoded_targets)

        x = F.relu(self.deconv_6(x, conv_2_out))
        x = self.up6(x, encoded_targets)

        x = F.relu(self.deconv_7(x, conv_1_out))
        x = self.up7(x, encoded_targets)
        
        output = torch.tanh(self.last(x))

        return output


class AdvP2P(nn.Module):
    def __init__(self, model_size=64, num_channels=8, upsample=True):
        super(AdvP2P, self).__init__()
        self.model_size = model_size  
        self.num_channels = num_channels  

        # downsample ans upsamples are changed
        latent_dim = model_size * 2
        self.linear_target_embedding = TargetEmbedding(5, latent_dim)

        stride = 1
        upsample = 5

        self.up1 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim, down=False)
        self.up2 = Block(model_size * 3, model_size * 3, laten_emb=latent_dim, down=False)
        self.up3 = Block(model_size, model_size, laten_emb=latent_dim, down=False)
        self.up5 = Block(model_size // 2, model_size // 2, laten_emb=latent_dim, down=False)
        self.up6 = Block(model_size // 4, model_size // 4, laten_emb=latent_dim, down=False)
        self.up7 = Block(num_channels, num_channels, laten_emb=latent_dim, down=False)

        self.deconv_1 = Upsample1DLayer(5 * model_size , 5 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Upsample1DLayerMulti(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Upsample1DLayerMulti(3* model_size * 2,  model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Upsample1DLayerMulti(model_size * 2, model_size // 2, 25, stride, upsample=2)
        self.deconv_6 = Upsample1DLayerMulti(model_size // 2 * 2 , model_size // 4, 25, stride, upsample=2)
        self.deconv_7 = Upsample1DLayerMulti(model_size // 4 * 2, num_channels, 25, stride, upsample=2)

        self.last = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=25, padding='same'))

        # new convolutional layers
        self.down1 = Block(num_channels, model_size // 4, laten_emb=latent_dim)
        self.down2 = Block(model_size // 4, model_size // 2, laten_emb=latent_dim)
        self.down3 = Block(model_size // 2, model_size, laten_emb=latent_dim)
        self.down4 = Block(model_size, model_size * 3, laten_emb=latent_dim)
        self.down5 = Block(model_size * 3, model_size * 5, laten_emb=latent_dim)
        self.down6 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim)

        self.conv_1 = nn.Conv1d(model_size // 4, model_size // 4, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 2, model_size // 2, 25, stride=2, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(model_size, model_size , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(model_size * 3, model_size * 3 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):

        encoded_targets = self.linear_target_embedding(targets)
        
        conv_1_out = self.down1(x, encoded_targets)
        conv_1_out = F.leaky_relu(self.conv_1(conv_1_out))
        
        conv_2_out = self.down2(conv_1_out, encoded_targets)
        conv_2_out = F.leaky_relu(self.conv_2(conv_2_out))
       
        conv_3_out = self.down3(conv_2_out, encoded_targets)
        conv_3_out = F.leaky_relu(self.conv_3(conv_3_out))
     
        conv_4_out = self.down4(conv_3_out, encoded_targets)
        conv_4_out = F.leaky_relu(self.conv_4(conv_4_out))
    
        conv_5_out = self.down5(conv_4_out, encoded_targets)
        conv_5_out = F.leaky_relu(self.conv_5(conv_5_out))
     
        conv_6_out = self.down6(conv_5_out, encoded_targets)
        x = F.leaky_relu(self.conv_6(conv_6_out))
      
        x = F.relu(self.deconv_1(x))
        x = self.up1(x, encoded_targets)

        x = F.relu(self.deconv_2(x, conv_5_out))
        x = self.up2(x, encoded_targets)
       
        x = F.relu(self.deconv_3(x, conv_4_out))
        x = self.up3(x, encoded_targets)
        
        x = F.relu(self.deconv_5(x, conv_3_out))
        x = self.up5(x, encoded_targets)

        x = F.relu(self.deconv_6(x, conv_2_out))
        x = self.up6(x, encoded_targets)

        x = F.relu(self.deconv_7(x, conv_1_out))
        x = self.up7(x, encoded_targets)
        
        output = torch.tanh(self.last(x))

        return output



class AdvP2PAutoEmb(nn.Module):
    def __init__(self, model_size=64, num_channels=8, upsample=True):
        super(AdvP2PAutoEmb, self).__init__()
        self.model_size = model_size  
        self.num_channels = num_channels  

        self.embedding = AutoPositionEmbedding()
        
        latent_dim = model_size * 2
        self.linear_target_embedding = TargetEmbedding(5, latent_dim)

        stride = 1
        upsample = 5

        self.up1 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim, down=False)
        self.up2 = Block(model_size * 3, model_size * 3, laten_emb=latent_dim, down=False)
        self.up3 = Block(model_size, model_size, laten_emb=latent_dim, down=False)
        self.up5 = Block(model_size // 2, model_size // 2, laten_emb=latent_dim, down=False)
        self.up6 = Block(model_size // 4, model_size // 4, laten_emb=latent_dim, down=False)
        self.up7 = Block(num_channels, num_channels, laten_emb=latent_dim, down=False)

        self.deconv_1 = Upsample1DLayer(5 * model_size , 5 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Upsample1DLayerMulti(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Upsample1DLayerMulti(3* model_size * 2,  model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Upsample1DLayerMulti(model_size * 2, model_size // 2, 25, stride, upsample=2)
        self.deconv_6 = Upsample1DLayerMulti(model_size // 2 * 2 , model_size // 4, 25, stride, upsample=2)
        self.deconv_7 = Upsample1DLayerMulti(model_size // 4 * 2, num_channels, 25, stride, upsample=2)

        self.last = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=25, padding='same'))

        # new convolutional layers
        self.down1 = Block(num_channels, model_size // 4, laten_emb=latent_dim)
        self.down2 = Block(model_size // 4, model_size // 2, laten_emb=latent_dim)
        self.down3 = Block(model_size // 2, model_size, laten_emb=latent_dim)
        self.down4 = Block(model_size, model_size * 3, laten_emb=latent_dim)
        self.down5 = Block(model_size * 3, model_size * 5, laten_emb=latent_dim)
        self.down6 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim)

        self.conv_1 = nn.Conv1d(model_size // 4, model_size // 4, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 2, model_size // 2, 25, stride=2, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(model_size, model_size , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(model_size * 3, model_size * 3 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):
        x = self.embedding(x)

        encoded_targets = self.linear_target_embedding(targets)
        
        conv_1_out = self.down1(x, encoded_targets)
        conv_1_out = F.leaky_relu(self.conv_1(conv_1_out))
        
        conv_2_out = self.down2(conv_1_out, encoded_targets)
        conv_2_out = F.leaky_relu(self.conv_2(conv_2_out))
       
        conv_3_out = self.down3(conv_2_out, encoded_targets)
        conv_3_out = F.leaky_relu(self.conv_3(conv_3_out))
     
        conv_4_out = self.down4(conv_3_out, encoded_targets)
        conv_4_out = F.leaky_relu(self.conv_4(conv_4_out))
    
        conv_5_out = self.down5(conv_4_out, encoded_targets)
        conv_5_out = F.leaky_relu(self.conv_5(conv_5_out))
     
        conv_6_out = self.down6(conv_5_out, encoded_targets)
        x = F.leaky_relu(self.conv_6(conv_6_out))
      
        x = F.relu(self.deconv_1(x))
        x = self.up1(x, encoded_targets)

        x = F.relu(self.deconv_2(x, conv_5_out))
        x = self.up2(x, encoded_targets)
       
        x = F.relu(self.deconv_3(x, conv_4_out))
        x = self.up3(x, encoded_targets)
        
        x = F.relu(self.deconv_5(x, conv_3_out))
        x = self.up5(x, encoded_targets)

        x = F.relu(self.deconv_6(x, conv_2_out))
        x = self.up6(x, encoded_targets)

        x = F.relu(self.deconv_7(x, conv_1_out))
        x = self.up7(x, encoded_targets)
        
        output = torch.tanh(self.last(x))

        return output



class AdvP2PPosEmb(nn.Module):
    def __init__(self, model_size=64, num_channels=8, upsample=True):
        super(AdvP2PAutoEmb, self).__init__()
        self.model_size = model_size  
        self.num_channels = num_channels  

        self.embedding = PositionalEmbedding()
        
        latent_dim = model_size * 2
        self.linear_target_embedding = TargetEmbedding(5, latent_dim)

        stride = 1
        upsample = 5

        self.up1 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim, down=False)
        self.up2 = Block(model_size * 3, model_size * 3, laten_emb=latent_dim, down=False)
        self.up3 = Block(model_size, model_size, laten_emb=latent_dim, down=False)
        self.up5 = Block(model_size // 2, model_size // 2, laten_emb=latent_dim, down=False)
        self.up6 = Block(model_size // 4, model_size // 4, laten_emb=latent_dim, down=False)
        self.up7 = Block(num_channels, num_channels, laten_emb=latent_dim, down=False)

        self.deconv_1 = Upsample1DLayer(5 * model_size , 5 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Upsample1DLayerMulti(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Upsample1DLayerMulti(3* model_size * 2,  model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Upsample1DLayerMulti(model_size * 2, model_size // 2, 25, stride, upsample=2)
        self.deconv_6 = Upsample1DLayerMulti(model_size // 2 * 2 , model_size // 4, 25, stride, upsample=2)
        self.deconv_7 = Upsample1DLayerMulti(model_size // 4 * 2, num_channels, 25, stride, upsample=2)

        self.last = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=25, padding='same'))

        # new convolutional layers
        self.down1 = Block(num_channels, model_size // 4, laten_emb=latent_dim)
        self.down2 = Block(model_size // 4, model_size // 2, laten_emb=latent_dim)
        self.down3 = Block(model_size // 2, model_size, laten_emb=latent_dim)
        self.down4 = Block(model_size, model_size * 3, laten_emb=latent_dim)
        self.down5 = Block(model_size * 3, model_size * 5, laten_emb=latent_dim)
        self.down6 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim)

        self.conv_1 = nn.Conv1d(model_size // 4, model_size // 4, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 2, model_size // 2, 25, stride=2, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(model_size, model_size , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(model_size * 3, model_size * 3 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):
        x = self.embedding(x)

        encoded_targets = self.linear_target_embedding(targets)
        
        conv_1_out = self.down1(x, encoded_targets)
        conv_1_out = F.leaky_relu(self.conv_1(conv_1_out))
        
        conv_2_out = self.down2(conv_1_out, encoded_targets)
        conv_2_out = F.leaky_relu(self.conv_2(conv_2_out))
       
        conv_3_out = self.down3(conv_2_out, encoded_targets)
        conv_3_out = F.leaky_relu(self.conv_3(conv_3_out))
     
        conv_4_out = self.down4(conv_3_out, encoded_targets)
        conv_4_out = F.leaky_relu(self.conv_4(conv_4_out))
    
        conv_5_out = self.down5(conv_4_out, encoded_targets)
        conv_5_out = F.leaky_relu(self.conv_5(conv_5_out))
     
        conv_6_out = self.down6(conv_5_out, encoded_targets)
        x = F.leaky_relu(self.conv_6(conv_6_out))
      
        x = F.relu(self.deconv_1(x))
        x = self.up1(x, encoded_targets)

        x = F.relu(self.deconv_2(x, conv_5_out))
        x = self.up2(x, encoded_targets)
       
        x = F.relu(self.deconv_3(x, conv_4_out))
        x = self.up3(x, encoded_targets)
        
        x = F.relu(self.deconv_5(x, conv_3_out))
        x = self.up5(x, encoded_targets)

        x = F.relu(self.deconv_6(x, conv_2_out))
        x = self.up6(x, encoded_targets)

        x = F.relu(self.deconv_7(x, conv_1_out))
        x = self.up7(x, encoded_targets)
        
        output = torch.tanh(self.last(x))

        return output
    


class AdvP2PNtFAutoEmb(nn.Module):
    def __init__(self, model_size=64, num_channels=8, upsample=True):
        super(AdvP2PAutoEmb, self).__init__()
        self.model_size = model_size  
        self.num_channels = num_channels  

        self.map_noise = nn.Sequential(nn.Conv1d(num_channels, 512, kernel_size=1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(512, 512, kernel_size=1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(512, num_channels, kernel_size=1),
                                        nn.LeakyReLU(0.2)
                                        )

        self.embedding = AutoPositionEmbedding()
        
        latent_dim = model_size * 2
        self.linear_target_embedding = TargetEmbedding(5, latent_dim)

        stride = 1
        upsample = 5

        self.up1 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim, down=False)
        self.up2 = Block(model_size * 3, model_size * 3, laten_emb=latent_dim, down=False)
        self.up3 = Block(model_size, model_size, laten_emb=latent_dim, down=False)
        self.up5 = Block(model_size // 2, model_size // 2, laten_emb=latent_dim, down=False)
        self.up6 = Block(model_size // 4, model_size // 4, laten_emb=latent_dim, down=False)
        self.up7 = Block(num_channels, num_channels, laten_emb=latent_dim, down=False)

        self.deconv_1 = Upsample1DLayer(5 * model_size , 5 * model_size, 25, stride, upsample=upsample)
        self.deconv_2 = Upsample1DLayerMulti(5 * model_size * 2, 3 * model_size, 25, stride, upsample=upsample)
        self.deconv_3 = Upsample1DLayerMulti(3* model_size * 2,  model_size, 25, stride, upsample=upsample)
        self.deconv_5 = Upsample1DLayerMulti(model_size * 2, model_size // 2, 25, stride, upsample=2)
        self.deconv_6 = Upsample1DLayerMulti(model_size // 2 * 2 , model_size // 4, 25, stride, upsample=2)
        self.deconv_7 = Upsample1DLayerMulti(model_size // 4 * 2, num_channels, 25, stride, upsample=2)

        self.last = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=25, padding='same'))

        # new convolutional layers
        self.down1 = Block(num_channels, model_size // 4, laten_emb=latent_dim)
        self.down2 = Block(model_size // 4, model_size // 2, laten_emb=latent_dim)
        self.down3 = Block(model_size // 2, model_size, laten_emb=latent_dim)
        self.down4 = Block(model_size, model_size * 3, laten_emb=latent_dim)
        self.down5 = Block(model_size * 3, model_size * 5, laten_emb=latent_dim)
        self.down6 = Block(model_size * 5, model_size * 5, laten_emb=latent_dim)

        self.conv_1 = nn.Conv1d(model_size // 4, model_size // 4, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(model_size // 2, model_size // 2, 25, stride=2, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(model_size, model_size , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(model_size * 3, model_size * 3 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(model_size * 5, model_size * 5 , 25, stride=5, padding= 25 // 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, targets):

        x = self.map_noise(x)

        x = self.embedding(x)

        encoded_targets = self.linear_target_embedding(targets)
        
        conv_1_out = self.down1(x, encoded_targets)
        conv_1_out = F.leaky_relu(self.conv_1(conv_1_out))
        
        conv_2_out = self.down2(conv_1_out, encoded_targets)
        conv_2_out = F.leaky_relu(self.conv_2(conv_2_out))
       
        conv_3_out = self.down3(conv_2_out, encoded_targets)
        conv_3_out = F.leaky_relu(self.conv_3(conv_3_out))
     
        conv_4_out = self.down4(conv_3_out, encoded_targets)
        conv_4_out = F.leaky_relu(self.conv_4(conv_4_out))
    
        conv_5_out = self.down5(conv_4_out, encoded_targets)
        conv_5_out = F.leaky_relu(self.conv_5(conv_5_out))
     
        conv_6_out = self.down6(conv_5_out, encoded_targets)
        x = F.leaky_relu(self.conv_6(conv_6_out))
      
        x = F.relu(self.deconv_1(x))
        x = self.up1(x, encoded_targets)

        x = F.relu(self.deconv_2(x, conv_5_out))
        x = self.up2(x, encoded_targets)
       
        x = F.relu(self.deconv_3(x, conv_4_out))
        x = self.up3(x, encoded_targets)
        
        x = F.relu(self.deconv_5(x, conv_3_out))
        x = self.up5(x, encoded_targets)

        x = F.relu(self.deconv_6(x, conv_2_out))
        x = self.up6(x, encoded_targets)

        x = F.relu(self.deconv_7(x, conv_1_out))
        x = self.up7(x, encoded_targets)
        
        output = torch.tanh(self.last(x))

        return output