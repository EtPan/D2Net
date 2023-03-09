import torch
import torch.nn as nn
import numpy as np
#from block3d import ConvBlock, DeconvBlock
from .block3d import ConvBlock, DeconvBlock
from pytorch_wavelets import DWTInverse,DWTForward


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class PMCB(nn.Module):
    def __init__(self, in_channels=64, d_rate=0.25, bias=True,negative_slope=0.05):
        super(PMCB, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3,1,1, bias=bias),
                                   nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(self.r_nc, in_channels, (3,1,1),1,(1,0,0), bias=bias),
                                   nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv3d(self.r_nc, in_channels, (1,3,3),1,(0,1,1), bias=bias),
                                   nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv3d(self.r_nc, self.d_nc, 1,1,0, bias=bias))
        
        self.c5 = nn.Conv3d(in_channels, in_channels, 1,1,0)
        
        self.ca = CALayer(self.d_nc*4)
        
    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.c5(self.ca(torch.cat((d1, d2, d3, d4), dim=1))) + x
        return res

class SCRB(nn.Module):
    def __init__(self, n_feats, bias=True, act=nn.ReLU(True)):
        super(SCRB, self).__init__()
        
        self.p3d = nn.Sequential(nn.Conv3d(1, n_feats, (3,1,1),1,(1,0,0), bias=bias),
                                  nn.Conv3d(n_feats, 1, (1,3,3),1,(0,1,1), bias=bias),
                                  act)
        self.p3d2 = nn.Sequential(nn.Conv3d(2, n_feats, (3,1,1),1,(1,0,0), bias=bias),
                                  nn.Conv3d(n_feats, 1, (1,3,3),1,(0,1,1), bias=bias),
                                  act)
        self.c1 = nn.Conv3d(3, 1, 1,1,0)                     

    def forward(self, x):
        x1 = self.p3d(x.unsqueeze(1))
        x2 = self.p3d2(torch.cat((x.unsqueeze(1),x1),dim=1))
        x3 = self.c1(torch.cat((x.unsqueeze(1),x1,x2),dim=1))+ x.unsqueeze(1)
        return x3.squeeze(1)

class D2NET(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_rcab, act_type='prelu',
                 norm_type=None):
        super(D2NET, self).__init__()

        self.num_features = num_features

        # LR feature extraction block
        self.conv_l = nn.Sequential(ConvBlock(in_channels, 4 * num_features, kernel_size=3,act_type=act_type, norm_type=norm_type),
                                     ConvBlock(4 * num_features, in_channels*3, kernel_size=1, act_type=act_type,norm_type=norm_type))

        self.conv_h = nn.ModuleList([
            nn.Sequential(ConvBlock(in_channels*3*2, 4 * num_features, kernel_size=3, act_type=act_type, norm_type=norm_type),
                          ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)),

            nn.Sequential(ConvBlock(in_channels*3*2, 4 * num_features, kernel_size=3, act_type=None, norm_type=norm_type),
                          DeconvBlock(4 * num_features, num_features, kernel_size=(1,2,2), stride=(1,2,2), act_type=act_type, norm_type=norm_type))])

        # recurrent block
        PMCBs = [PMCB(num_features) for _ in range(num_rcab)]
        self.rbh = nn.Sequential(*PMCBs)
        
        self.trans = nn.Conv3d(num_features, in_channels*3, 1,1,0) 

        # reconstruction block
        self.conv_steps = nn.ModuleList([
            nn.Sequential(ConvBlock(in_channels*3, num_features, kernel_size=3, act_type=act_type, norm_type=norm_type),
                          ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)),

            nn.Sequential(ConvBlock(num_features, num_features, kernel_size=3, act_type=act_type, norm_type=norm_type),
                          ConvBlock(num_features, out_channels * 3, kernel_size=3, act_type=None, norm_type=norm_type)),

            nn.Sequential(ConvBlock(num_features, num_features, kernel_size=1, stride=(1,2,2), act_type=act_type, norm_type=norm_type),
                          ConvBlock(num_features, out_channels * 3, kernel_size=3, act_type=None, norm_type=norm_type))])

        self.DWT = DWTForward(J=2, wave='haar').cuda() 
        self.IDWT = DWTInverse(wave='haar').cuda()
        
        # fine b
        self.ResB = SCRB(64)

    def forward(self, x):

        xl,xh = self.DWT(x.squeeze(1))

        xl = xl.unsqueeze(1)
        xl = self.conv_l(xl)
        Yl = self.conv_steps[0](xl)
        Yl = Yl.squeeze(1)

        xh = [xh[i].transpose(2, 1) for i in range(len(xh))]

        xh1 = self.conv_h[1](torch.cat((xh[1],xl),dim=1))
        hh1 = self.rbh(xh1)
        yh1 = self.conv_steps[1](hh1)
        
        xh0 = self.conv_h[0](torch.cat((xh[0],self.trans(hh1)),dim=1))
        hh0 = self.rbh(xh0)
        yh0 = self.conv_steps[-1](hh0)
        
        Yh = [yh1.transpose(2, 1),yh0.transpose(2, 1)]

        idwt2 = self.IDWT((Yl, Yh))
        
        output = self.ResB(idwt2).unsqueeze(1)

        return output

if __name__ == '__main__':
    from torchsummary import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = D2NET(in_channels=1, out_channels=1, num_features=32, num_rcab=4).to(device)
    #print(net)
    summary(net, (1, 31, 64, 64))