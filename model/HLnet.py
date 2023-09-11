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

        self.fc1 = nn.Conv2d(input_nc, input_nc // ratio, 1, bias=True)
        # self.bn1 = nn.BatchNorm2d(input_nc // ratio)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(input_nc // ratio, input_nc, 1, bias=True)
        # self.bn2 = nn.BatchNorm2d(input_nc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.bn2(self.fc2(self.relu1(self.bn1(self.fc1(self.avg_pool(x))))))
        # max_out = self.bn2(self.fc2(self.relu1(self.bn1(self.fc1(self.max_pool(x))))))
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

class basicConv(nn.Module):
    def __init__(self, in_channels=192, out_channels=128, kernel_size=3, padding=1,stride=1):
        super(basicConv, self).__init__()
        if stride==1:
            self.conv=nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        elif stride>1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    def forward(self,x):
        out=self.conv(x)
        return out




class HLCDNetNo(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(HLCDNetNo, self).__init__()
        # sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        # self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))),
        #                                    requires_grad=False)
        # sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        # self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))),
        #                                    requires_grad=False)
        self.Lmoudle=nn.Sequential(
            basicConv(in_channels=3,out_channels=16,kernel_size=7,padding=3),
            # basicConv(in_channels=16, out_channels=16, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#128
            basicConv(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            # basicConv(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#64
            basicConv(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            # basicConv(in_channels=64, out_channels=64, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32
            basicConv(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            basicConv(in_channels=128, out_channels=128, kernel_size=7, padding=3),
            ChannelAttention(input_nc=128, ratio=8),
            nn.BatchNorm2d(128),
        )
        self.backboneH = resnet34_2l(pretrained=True, replace_stride_with_dilation=[False, False, True])  # resnet34
        self.BasicBlockHLd1 = nn.Sequential(
            BasicBlock(128, 128),
            # BasicBlock(128, 128)
        )
        self.deconv_1 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            BasicBlock(128, 128)
            # basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.deconv_2 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64),
            # basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),

        )
        self.deconv_3=nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64),
            BasicBlock(64, 32)
            # basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # basicConv(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        )
        self.deconv_classier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
        )
    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        # L = input_data[:, 3, :, :]
        # L = L.unsqueeze(1)
        # # input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]
        #
        # edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        # edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        # edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        # # weight_edge = inputData * (1 + edge)  # 高频
        # weight_edge=inputData+edge
        # weight_edge=self.sigmoid1(weight_edge)
        # blur = F.conv2d(inputData, self.blur_weight, stride=1, padding=1, groups=3)
        # blur=self.sigmoid2(blur)

        return inputData, inputData

    def extractorH(self, input1,input2):
        fea1 = self.backboneH.conv1(input1)
        fea1 = self.backboneH.bn1(fea1)
        fea1 = self.backboneH.relu(fea1)
        featH1_0 = self.backboneH.maxpool(fea1) #torch.Size([60, 64, 64, 64])
        featH1_1 = self.backboneH.layer1(featH1_0)#[60, 64, 64, 64]
        featH1_2 = self.backboneH.layer2(featH1_1)#[60, 128, 32, 32]

        fea2 = self.backboneH.conv1(input2)
        fea2 = self.backboneH.bn1(fea2)
        fea2 = self.backboneH.relu(fea2)
        featH2_0 = self.backboneH.maxpool(fea2)#torch.Size([60, 64, 64, 64])
        featH2_1 = self.backboneH.layer1(featH2_0)#[60, 64, 64, 64]
        featH2_2 = self.backboneH.layer2(featH2_1)#[60, 128, 32, 32]

        # for i in [featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2]:
        #     print(i.shape)
        return fea1, fea2, featH1_1, featH2_1, featH1_2, featH2_2

    def extractorL(self, input1,input2):
        L_AC1 = self.Lmoudle(input1)
        L_AC2 = self.Lmoudle(input2)#[60, 128, 1, 1]
        return L_AC1,L_AC2


    def forward(self, pre_data, post_data):
        inputH1, inputL1 = self.filterHL(pre_data)
        inputH2, inputL2 = self.filterHL(post_data)
        L_AC1, L_AC2 = self.extractorL(inputL1,inputL2)#[60, 128, 1, 1]
        L_AC = (L_AC1 + L_AC2) / 2

        featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2 = self.extractorH(inputH1,inputH2) #torch.Size([60, 64, 64, 64]) #[60, 128, 32, 32]
        CAW1 = featH1_2 * (1+L_AC)
        CAW2 = featH2_2 * (1+L_AC)#[60, 128, 32, 32]
        Fusion1_1=self.BasicBlockHLd1(CAW1)#([60, 128, 32, 32])
        Fusion1_2=self.BasicBlockHLd1(CAW2)#([60, 128, 32, 32])
        FusionFeat=torch.cat([Fusion1_1,Fusion1_2],dim=1)#([60, 256, 32, 32])
        defeat1=self.deconv_1(FusionFeat)#([60, 128, 64, 64]
        diffFeat1=torch.abs(featH1_1- featH2_1)
        defeat1_cat=torch.cat([defeat1,diffFeat1],dim=1)#60, 192, 64, 64
        defeat2=self.deconv_2(defeat1_cat)
        diffFeat2 = torch.abs(featH1_0 - featH2_0)
        defeat2_cat = torch.cat([defeat2, diffFeat2], dim=1)  #[60, 128, 128, 128]
        defeat3=self.deconv_3(defeat2_cat)
        pre=self.deconv_classier(defeat3)
        # print('pre',pre.shape,CAW2.shape)
        # self.encoder(post_data)
        return pre,pre

    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()



class HLCDNetG(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(HLCDNetG, self).__init__()
        # sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        # self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))),
        #                                    requires_grad=False)
        # sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        # self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))),
        #                                    requires_grad=False)
        gussiankernal = self.get_gaussian_kernel2()
        self.blur_weight = nn.Parameter(torch.from_numpy(gussiankernal), requires_grad=False)
        self.Lmoudle = nn.Sequential(
            basicConv(in_channels=3, out_channels=16, kernel_size=7, padding=3),
            # basicConv(in_channels=16, out_channels=16, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 128
            basicConv(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            # basicConv(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64
            basicConv(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # basicConv(in_channels=64, out_channels=64, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32
            basicConv(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            ChannelAttention(input_nc=128, ratio=8)
        )
        self.backboneH = resnet34_2l(pretrained=True, replace_stride_with_dilation=[False, False, True])  # resnet34
        self.BasicBlockHLd1 = nn.Sequential(
            BasicBlock(128, 128),
            # BasicBlock(128, 128)
        )
        self.deconv_1 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # BasicBlock(128, 128)
            basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.deconv_2 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # BasicBlock(64, 64),
            basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),

        )
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # BasicBlock(64, 64),
            # BasicBlock(64, 32)
            basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            basicConv(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        )
        self.deconv_classier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
        )

    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        # L = input_data[:, 3, :, :]
        # L = L.unsqueeze(1)
        # # input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]
        #
        # edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        # edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        # edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        # # weight_edge = inputData * (1 + edge)  # 高频
        # weight_edge=inputData+edge
        # weight_edge=self.sigmoid1(weight_edge)
        blur = F.conv2d(inputData, self.blur_weight, stride=1, padding=1, groups=3)
        # blur=self.sigmoid2(blur)

        return inputData, blur

    def extractorH(self, input1, input2):
        fea1 = self.backboneH.conv1(input1)
        fea1 = self.backboneH.bn1(fea1)
        fea1 = self.backboneH.relu(fea1)
        featH1_0 = self.backboneH.maxpool(fea1)  # torch.Size([60, 64, 64, 64])
        featH1_1 = self.backboneH.layer1(featH1_0)  # [60, 64, 64, 64]
        featH1_2 = self.backboneH.layer2(featH1_1)  # [60, 128, 32, 32]

        fea2 = self.backboneH.conv1(input2)
        fea2 = self.backboneH.bn1(fea2)
        fea2 = self.backboneH.relu(fea2)
        featH2_0 = self.backboneH.maxpool(fea2)  # torch.Size([60, 64, 64, 64])
        featH2_1 = self.backboneH.layer1(featH2_0)  # [60, 64, 64, 64]
        featH2_2 = self.backboneH.layer2(featH2_1)  # [60, 128, 32, 32]

        # for i in [featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2]:
        #     print(i.shape)
        return fea1, fea2, featH1_1, featH2_1, featH1_2, featH2_2

    def extractorL(self, input1, input2):
        L_AC1 = self.Lmoudle(input1)
        L_AC2 = self.Lmoudle(input2)  # [60, 128, 1, 1]
        return L_AC1, L_AC2

    def forward(self, pre_data, post_data):
        inputH1, inputL1 = self.filterHL(pre_data)
        inputH2, inputL2 = self.filterHL(post_data)
        L_AC1, L_AC2 = self.extractorL(inputL1, inputL2)  # [60, 128, 1, 1]
        L_AC = (L_AC1 + L_AC2) / 2
        featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2 = self.extractorH(inputH1,
                                                                                     inputH2)  # torch.Size([60, 64, 64, 64]) #[60, 128, 32, 32]
        CAW1 = featH1_2 * (1 + L_AC)
        CAW2 = featH2_2 * (1 + L_AC)  # [60, 128, 32, 32]
        Fusion1_1 = self.BasicBlockHLd1(CAW1)  # ([60, 128, 32, 32])
        Fusion1_2 = self.BasicBlockHLd1(CAW2)  # ([60, 128, 32, 32])
        FusionFeat = torch.cat([Fusion1_1, Fusion1_2], dim=1)  # ([60, 256, 32, 32])
        defeat1 = self.deconv_1(FusionFeat)  # ([60, 128, 64, 64]
        diffFeat1 = torch.abs(featH1_1 - featH2_1)
        defeat1_cat = torch.cat([defeat1, diffFeat1], dim=1)  # 60, 192, 64, 64
        defeat2 = self.deconv_2(defeat1_cat)
        diffFeat2 = torch.abs(featH1_0 - featH2_0)
        defeat2_cat = torch.cat([defeat2, diffFeat2], dim=1)  # [60, 128, 128, 128]
        defeat3 = self.deconv_3(defeat2_cat)
        pre = self.deconv_classier(defeat3)
        # print('pre',pre.shape,CAW2.shape)
        # self.encoder(post_data)
        return pre, pre

    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

    def get_gaussian_kernel(self, kernel_size=3, sigma=1, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                    groups=channels, bias=False, padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def get_gaussian_kernel2(self, kernel_size=3, sigma=1, channels=3):
        kernel = np.zeros(shape=(kernel_size, kernel_size), dtype=np.float32)
        radius = kernel_size // 2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                # 二维高斯函数
                v = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
        kernel2 = kernel / np.sum(kernel)
        kernel2 = np.array(kernel2, dtype='float32').reshape(1, 1, 3, 3)
        # kernel2=np.repeat(kernel2,3,axis=1)
        kernel2 = np.repeat(kernel2, 3, axis=0)
        return kernel2

class HLCDNetGB(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(HLCDNetGB, self).__init__()
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))),
                                           requires_grad=False)
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))),
                                           requires_grad=False)
        gussiankernal = self.get_gaussian_kernel2()
        self.blur_weight = nn.Parameter(torch.from_numpy(gussiankernal), requires_grad=False)
        self.Lmoudle = nn.Sequential(
            basicConv(in_channels=3, out_channels=16, kernel_size=7, padding=3),
            # basicConv(in_channels=16, out_channels=16, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 128
            basicConv(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            # basicConv(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64
            basicConv(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # basicConv(in_channels=64, out_channels=64, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32
            basicConv(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            ChannelAttention(input_nc=128, ratio=8)
        )
        self.backboneH = resnet34_2l(pretrained=True, replace_stride_with_dilation=[False, False, True])  # resnet34
        self.BasicBlockHLd1 = nn.Sequential(
            BasicBlock(128, 128),
            # BasicBlock(128, 128)
        )
        self.deconv_1 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # BasicBlock(128, 128)
            basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.deconv_2 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # BasicBlock(64, 64),
            basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),

        )
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # BasicBlock(64, 64),
            # BasicBlock(64, 32)
            basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            basicConv(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        )
        self.deconv_classier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
        )

    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        L = input_data[:, 3, :, :]
        L = L.unsqueeze(1)
        # input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]

        edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        # weight_edge = inputData * (1 + edge)  # 高频
        weight_edge=inputData+edge
        # weight_edge=self.sigmoid1(weight_edge)
        blur = F.conv2d(inputData, self.blur_weight, stride=1, padding=1, groups=3)
        # blur=self.sigmoid2(blur)

        return weight_edge, blur

    def extractorH(self, input1, input2):
        fea1 = self.backboneH.conv1(input1)
        fea1 = self.backboneH.bn1(fea1)
        fea1 = self.backboneH.relu(fea1)
        featH1_0 = self.backboneH.maxpool(fea1)  # torch.Size([60, 64, 64, 64])
        featH1_1 = self.backboneH.layer1(featH1_0)  # [60, 64, 64, 64]
        featH1_2 = self.backboneH.layer2(featH1_1)  # [60, 128, 32, 32]

        fea2 = self.backboneH.conv1(input2)
        fea2 = self.backboneH.bn1(fea2)
        fea2 = self.backboneH.relu(fea2)
        featH2_0 = self.backboneH.maxpool(fea2)  # torch.Size([60, 64, 64, 64])
        featH2_1 = self.backboneH.layer1(featH2_0)  # [60, 64, 64, 64]
        featH2_2 = self.backboneH.layer2(featH2_1)  # [60, 128, 32, 32]

        # for i in [featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2]:
        #     print(i.shape)
        return fea1, fea2, featH1_1, featH2_1, featH1_2, featH2_2

    def extractorL(self, input1, input2):
        L_AC1 = self.Lmoudle(input1)
        L_AC2 = self.Lmoudle(input2)  # [60, 128, 1, 1]
        return L_AC1, L_AC2

    def forward(self, pre_data, post_data):
        inputH1, inputL1 = self.filterHL(pre_data)
        inputH2, inputL2 = self.filterHL(post_data)
        L_AC1, L_AC2 = self.extractorL(inputL1, inputL2)  # [60, 128, 1, 1]
        L_AC = (L_AC1 + L_AC2) / 2
        featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2 = self.extractorH(inputH1,
                                                                                     inputH2)  # torch.Size([60, 64, 64, 64]) #[60, 128, 32, 32]
        CAW1 = featH1_2 * (1 + L_AC)
        CAW2 = featH2_2 * (1 + L_AC)  # [60, 128, 32, 32]
        Fusion1_1 = self.BasicBlockHLd1(CAW1)  # ([60, 128, 32, 32])
        Fusion1_2 = self.BasicBlockHLd1(CAW2)  # ([60, 128, 32, 32])
        FusionFeat = torch.cat([Fusion1_1, Fusion1_2], dim=1)  # ([60, 256, 32, 32])
        defeat1 = self.deconv_1(FusionFeat)  # ([60, 128, 64, 64]
        diffFeat1 = torch.abs(featH1_1 - featH2_1)
        defeat1_cat = torch.cat([defeat1, diffFeat1], dim=1)  # 60, 192, 64, 64
        defeat2 = self.deconv_2(defeat1_cat)
        diffFeat2 = torch.abs(featH1_0 - featH2_0)
        defeat2_cat = torch.cat([defeat2, diffFeat2], dim=1)  # [60, 128, 128, 128]
        defeat3 = self.deconv_3(defeat2_cat)
        pre = self.deconv_classier(defeat3)
        # print('pre',pre.shape,CAW2.shape)
        # self.encoder(post_data)
        return pre, pre

    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

    def get_gaussian_kernel(self, kernel_size=3, sigma=1, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                    groups=channels, bias=False, padding=kernel_size // 2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def get_gaussian_kernel2(self, kernel_size=3, sigma=1, channels=3):
        kernel = np.zeros(shape=(kernel_size, kernel_size), dtype=np.float32)
        radius = kernel_size // 2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                # 二维高斯函数
                v = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
        kernel2 = kernel / np.sum(kernel)
        kernel2 = np.array(kernel2, dtype='float32').reshape(1, 1, 3, 3)
        # kernel2=np.repeat(kernel2,3,axis=1)
        kernel2 = np.repeat(kernel2, 3, axis=0)
        return kernel2