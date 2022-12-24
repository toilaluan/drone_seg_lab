import torch 
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class ASPP(nn.Module):
    
    def __init__(self, in_channels=2048, out_channels=256, atrous=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.block1 = BasicBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                 stride=1, padding=0, dilation=1, bias=True)
        self.block2 = BasicBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                 stride=1, padding="same", dilation=atrous[0], bias=True)
        self.block3 = BasicBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                 stride=1, padding="same", dilation=atrous[1], bias=True)
        self.block4 = BasicBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                 stride=1, padding="same", dilation=atrous[2], bias=True)
        self.imgblock = BasicBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                 stride=1, padding="same", dilation=1, bias=True)
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.final_block = BasicBlock(in_channels=out_channels*5, out_channels=out_channels, kernel_size=1,
                                     stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        imf = self.gavg(x)
        imf = self.imgblock(imf)
        out5 = F.interpolate(imf, size=x.size()[2:], mode="bilinear", align_corners=True)
        out_concat = torch.concat([out1, out2, out3, out4, out5], axis=1)
        out = self.final_block(out_concat)
        return out