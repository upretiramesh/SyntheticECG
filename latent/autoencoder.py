import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample=None, final = False):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.1) if not final else nn.Tanh()
       
    def forward(self, x):
        out = self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        return self.act(self.bn(out))
   

class DownBlock(nn.Module):
    def __init__(self, ins, outs, k, s):
        super(DownBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv1d(ins, outs, kernel_size=k, stride=s, padding=k//2),
                    nn.BatchNorm1d(outs),
                    nn.LeakyReLU(0.1))
    def forward(self, x):
        return self.layer(x)



class AutoEncoder(nn.Module):
    def __init__(self, model_size=50, num_channels=8):
        super(AutoEncoder, self).__init__()
        self.model_size = model_size
        self.num_channels = num_channels 

        # Encode 
        self.conv_1 = DownBlock(num_channels, int(model_size / 5), 25, 2)
        self.conv_2 = DownBlock(model_size // 5, model_size // 2, 25, 2) 
        self.conv_3 = DownBlock(model_size // 2, model_size , 25, 2) 
        self.conv_4 = DownBlock(model_size, model_size * 3 , 25, 5) 
        self.conv_5 = DownBlock(model_size * 3, model_size * 5 , 25, 5) 
        self.conv_6 = DownBlock(model_size * 5, model_size * 5 , 5, 5) 
        self.last = nn.Sequential(
                                nn.Conv1d(model_size * 5, model_size * 10, 5),
                                nn.BatchNorm1d(model_size * 10), 
                                nn.LeakyReLU(0.1))

        # Decoder 
        upsample = 5

        self.deconv_0 = Transpose1dLayer(10 * model_size, 5 * model_size, 5, upsample=upsample) 
        self.deconv_1 = Transpose1dLayer(5 * model_size, 5 * model_size, 5, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(5 * model_size, 3 * model_size, 25, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(3 * model_size,  model_size, 25, upsample=upsample)
        self.deconv_5 = Transpose1dLayer(model_size, int(model_size / 2), 25, upsample=2)
        self.deconv_6 = Transpose1dLayer(int(model_size / 2), int(model_size / 5), 25, upsample=2)
        self.deconv_7 = Transpose1dLayer(int(model_size / 5), num_channels, 25, upsample=2, final=True)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, latent=False):

        conv_1_out = self.conv_1(x)
       
        conv_2_out = self.conv_2(conv_1_out)
       
        conv_3_out = self.conv_3(conv_2_out)
       
        conv_4_out = self.conv_4(conv_3_out)
       
        conv_5_out = self.conv_5(conv_4_out)
       
        conv_6_out = self.conv_6(conv_5_out)

        laten = self.last(conv_6_out)

        if latent:
            return torch.flatten(laten, start_dim=1)

        x = self.deconv_0(laten)
        
        x = self.deconv_1(x)
        
        x = self.deconv_2(x)

        x = self.deconv_3(x)
        
        x = self.deconv_5(x)
        
        x = self.deconv_6(x)

        output = self.deconv_7(x)

        return output
