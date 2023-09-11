from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.resnet import resnet34_2l
import matplotlib.pyplot as plt
class ChannelAttention(nn.Module):
    def __init__(self, input_nc, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(input_nc, input_nc // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(input_nc // ratio, input_nc, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
class ResNetpre(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet34', pretrained=True,Hflag=False):
        super(ResNetpre, self).__init__()
        self.Hflag=Hflag
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2

    ####18:
    # layer0  torch.Size([2, 64, 64, 64])
    # layer1  torch.Size([2, 64, 64, 64])
    # layer2  torch.Size([2, 128, 32, 32])
    # layer3  torch.Size([2, 256, 16, 16])  Total params: 2,782,784
    # layer4  torch.Size([2, 512, 16, 16])  Total params: 11,176,512
    ######34:
    # layer0   torch.Size([2, 64, 64, 64])
    # layer1   torch.Size([2, 64, 64, 64])
    # layer2    torch.Size([2, 128, 32, 32])
    # layer3   torch.Size([2, 256, 16, 16])  Total params: 8,170,304
    # layer4    torch.Size([2, 512, 16, 16]) Total params: 21,284,672
    def forward(self, x1):
        x11 = self.layer0(x1)#torch.Size([4, 64, 64, 64])
        # print('x11',x11.shape)
        x21 = self.layer1(x11)#torch.Size([4, 64, 64, 64])
        # print('x21', x21.shape)
        x31 = self.layer2(x21)#torch.Size([4, 128, 32, 32])
        # print('x31', x31.shape)
        return x11,x21, x31

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.convplus = nn.Conv2d(inplanes, planes, stride=stride,kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        residual = self.convplus(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out+=residual
        out=self.relu(out)
        return out