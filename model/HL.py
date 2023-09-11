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


class HLCDNetlog(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(HLCDNetlog, self).__init__()
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))),requires_grad=False)

        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))),requires_grad=False)
        gussiankernal = self.get_gaussian_kernel2()
        self.blur_weight = nn.Parameter(torch.from_numpy(gussiankernal),requires_grad=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        # self.backboneH = ResNetpre(in_channels=3, output_stride=2, pretrained=False,backbone='resnet34')
        # self.backboneL = ResNetpre(in_channels=3, output_stride=2, pretrained=False,backbone='resnet34')
        self.backboneH = resnet34_2l(pretrained=True,replace_stride_with_dilation=[False, False, True])#resnet34
        self.backboneL = resnet34_2l(pretrained=True,replace_stride_with_dilation=[False, False, True])

        # self.BasicBlockL1 = BasicBlock(128, 128)
        self.BasicBlockL2 = BasicBlock(256, 128)
        self.BasicBlockH1 = BasicBlock(64, 64)
        self.BasicBlockHLd1 = BasicBlock(128, 64)
        # self.max_pool_HL1 = nn.MaxPool2d(kernel_size=2)#32
        self.BasicBlockHLd2 = BasicBlock(64, 64)
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv_classier = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.up_H = nn.PixelShuffle(upscale_factor=2)
        self.up_L = nn.PixelShuffle(upscale_factor=2)

    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        L = input_data[:, 3, :, :]
        L = L.unsqueeze(1)
        # input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]

        edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        weight_edge = inputData * (1 + edge)  # 高频
        # weight_edge=inputData+edge
        # weight_edge=self.sigmoid1(weight_edge)
        blur = F.conv2d(inputData, self.blur_weight, stride=1, padding=1, groups=3)
        # blur=self.sigmoid2(blur)
        # for i in range(4):
        #     inputDataRGB=inputData[i,:,:,:].detach().cpu().numpy()
        #     inputDataRGB=inputDataRGB.transpose((1,2,0))
        #     image_edge=edge[i,0,:,:].detach().cpu().numpy()
        #     print('aaaa',image_edge.shape,inputDataRGB.shape)
        #     plt.figure('img_'+str(i))
        #     plt.imshow(inputDataRGB)
        #     plt.savefig('./testout/img_%d.png'%i)
        #     plt.figure('edge_' + str(i))
        #     plt.imshow(image_edge)
        #     plt.savefig('./testout/edge_%d.png' % i)
        #     #blur
        #     blurimg = blur[i, :, :, :].detach().cpu().numpy()
        #     blurimg = blurimg.transpose((1, 2, 0))
        #     print('blurimg',blurimg.shape,np.max(blurimg))
        #     plt.figure('blur_' + str(i))
        #     plt.imshow(blurimg)
        #     plt.savefig('./testout/blur_%d.png' % i)
        #
        #     weight_edge_img = weight_edge[i, :, :, :].detach().cpu().numpy()
        #     weight_edge_img = weight_edge_img.transpose((1, 2, 0))
        #     print('weight_edge', weight_edge_img.shape,np.max(weight_edge_img))
        #     plt.figure('weight_edge_' + str(i))
        #     plt.imshow(weight_edge_img)
        #     plt.savefig('./testout/weight_edge_%d.png' % i)
        #     # plt.show()
        # return weight_edge, blur

        return weight_edge, blur

    def extractorH(self, input1,input2):
        # featH1_0 = self.backboneH.layer0(input1)
        fea1 = self.backboneH.conv1(input1)
        fea1 = self.backboneH.bn1(fea1)
        fea1 = self.backboneH.relu(fea1)
        featH1_0 = self.backboneH.maxpool(fea1)
        featH1_1 = self.backboneH.layer1(featH1_0)
        refeatH1 = self.backboneH.layer2(featH1_1)

        fea2 = self.backboneH.conv1(input2)
        fea2 = self.backboneH.bn1(fea2)
        fea2 = self.backboneH.relu(fea2)
        featH2_0 = self.backboneH.maxpool(fea2)
        # featH2_0 = self.backboneH.layer0(input2)
        featH2_1 = self.backboneH.layer1(featH2_0)
        refeatH2 = self.backboneH.layer2(featH2_1)

        # featH1_0,featH1_1,refeatH1=self.backboneH(input1)#torch.Size([2, 64, 64, 64])  torch.Size([2, 128, 32, 32])
        # featH2_0,featH2_1, refeatH2 = self.backboneH(input2)  # torch.Size([2, 64, 64, 64]) torch.Size([2, 128, 32, 32])
        # featH1_1=self.BasicBlockH1(refeatH1)
        # featH2_1 = self.BasicBlockH1(refeatH2)#torch.Size([4, 64, 64, 64])
        # print('refeatH1', refeatH1.shape)
        return featH1_0,featH1_0,featH1_1,featH2_1,refeatH1,refeatH2
    def extractorL(self, input1,input2):
        fea1 = self.backboneL.conv1(input1)
        fea1 = self.backboneL.bn1(fea1)
        fea1 = self.backboneL.relu(fea1)
        refeatL1_0 = self.backboneL.maxpool(fea1)
        # refeatL1_0 = self.backboneL.layer0(input1)
        refeatL1_1 = self.backboneL.layer1(refeatL1_0)
        refeatL1 = self.backboneL.layer2(refeatL1_1)

        fea2 = self.backboneL.conv1(input2)
        fea2 = self.backboneL.bn1(fea2)
        fea2 = self.backboneL.relu(fea2)
        refeatL2_0 = self.backboneL.maxpool(fea2)
        # refeatL2_0 = self.backboneL.layer0(input2)
        refeatL2_1 = self.backboneL.layer1(refeatL2_0)
        refeatL2 = self.backboneL.layer2(refeatL2_1)
        # print('refeatL2',refeatL2.shape)
        # refeatL1_0,refeatL1_1, refeatL1 = self.backboneL(input1)#torch.Size([2, 64, 64, 64]) torch.Size([2, 128, 32, 32])
        # refeatL2_0,refeatL2_1, refeatL2 = self.backboneL(input2)  # torch.Size([2, 64, 64, 64]) torch.Size([2, 128, 32, 32])
        featLcat = torch.cat([refeatL1, refeatL2], dim=1)#torch.Size([2, 256, 32, 32])
        # featLcat1=self.BasicBlockL1(featLcat)#128
        featLcat2 = self.BasicBlockL2(featLcat)#torch.Size([2, 128, 32, 32])
        # print('featLcat2', featLcat2.shape)
        return refeatL1_0,refeatL2_0,refeatL1_1,refeatL2_1,refeatL1,refeatL2,featLcat2
    def HLencoderD(self,HL1,HL2):
        featHL1_1 = self.BasicBlockHLd1(HL1)
        featHL2_1 = self.BasicBlockHLd1(HL2)#64*32*32
        # featHL1_1pool = self.max_pool_HL1(featHL1_1)
        # featHL2_1pool = self.max_pool_HL1(featHL2_1)
        featHL1_2= self.BasicBlockHLd2(featHL1_1)#64*32*32
        featHL2_2 = self.BasicBlockHLd2(featHL2_1)
        return featHL1_1,featHL2_1,featHL1_2,featHL2_2


    def forward(self, pre_data, post_data):
        pre_dataLog = torch.log(pre_data+1)
        post_dataLog=torch.log(post_data+1)
        inputH1, inputL1 = self.filterHL(pre_dataLog)
        inputH2, inputL2 = self.filterHL(post_dataLog)
        featH1_0,featH2_0,featH1_1,featH2_1,refeatH1,refeatH2 = self.extractorH(inputH1,inputH2)#torch.Size([4, 64, 64, 64]) torch.Size([4, 128, 32, 32])
        featL1_0,featL2_0,featL1_1,featL2_1,refeatL1,refeatL2,refeatL = self.extractorL(inputL1,inputL2)#torch.Size([4, 64, 64, 64])  torch.Size([4, 128, 32, 32])
        # featHL1=torch.cat([refeatH1,refeatL],dim=1)#128
        # featHL2 = torch.cat([refeatH2, refeatL], dim=1)#torch.Size([4, 128, 64, 64])
        featHL1 = refeatH1 + refeatL#torch.Size([4, 128, 64, 64])
        featHL2 = refeatH2 + refeatL
        # print('featHL1', featHL1.shape)
        featHLDown1_l1,featHLDown2_l2,featHLDown1,featHLDown2=self.HLencoderD(featHL1,featHL2)#torch.Size([4, 64, 32, 32])
        # print('featHLDown1', featHLDown1.shape)
        featHLDown1Exp = torch.exp(featHLDown1)
        featHLDown2Exp = torch.exp(featHLDown2)
        featcat=torch.cat([featHLDown1Exp,featHLDown2Exp,torch.abs(featHLDown1Exp-featHLDown2Exp)],dim=1)#torch.Size([4, 192, 32, 32])
        # featcat=torch.cat([featHLDown1,featHLDown2,torch.abs(featHLDown1-featHLDown2)],dim=1)#torch.Size([4, 192, 32, 32])

        HLup1 = self.deconv_1(featcat)#torch.Size([4, 64, 64, 64])
        # print('HLup1',HLup1.shape)
        # HLup1_cat=torch.cat([torch.abs(torch.exp(featH1_1)-torch.exp(featH2_1)),
        #                      torch.abs(torch.exp(featL1_1)-torch.exp(featL2_1)),HLup1],dim=1)#torch.Size([4, 192, 64, 64])
        HLup1_cat = torch.cat([torch.abs(featH1_1 - featH2_1),
                               torch.abs(featL1_1 - featL2_1), HLup1],dim=1)  # torch.Size([4, 192, 64, 64])
        # print('HLup1_cat', HLup1_cat.shape)
        HLup2=self.deconv_2(HLup1_cat)#torch.Size([4, 64, 128, 128])
        # print('HLup2', HLup2.shape)
        diffL_0 = self.up_L(torch.abs(featL1_0 - featL2_0))#
        diffH_0 = self.up_H(torch.abs(featH1_0 - featH2_0))#torch.Size([4, 16, 128, 128])
        # print('diffL_0', diffL_0.shape)
        HLup2_cat = torch.cat([diffL_0,diffH_0, HLup2], dim=1)  # torch.Size([4, 96, 128, 128])
        pre=self.deconv_classier(HLup2_cat)
        # self.encoder(post_data)
        return pre,pre

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
        kernel = np.zeros(shape=(kernel_size, kernel_size), dtype=np.float)
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





class HLCDNet2(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(HLCDNet2, self).__init__()
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))),
                                           requires_grad=False)

        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))),
                                           requires_grad=False)
        gussiankernal = self.get_gaussian_kernel2()
        self.blur_weight = nn.Parameter(torch.from_numpy(gussiankernal),requires_grad=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.backboneH = resnet34_2l(pretrained=True,replace_stride_with_dilation=[False, False, True])#resnet34
        self.backboneL = resnet34_2l(pretrained=True,replace_stride_with_dilation=[False, False, True])

        self.ca_L1 = ChannelAttention(input_nc=64, ratio=8)
        self.ca_L2 = ChannelAttention(input_nc=128, ratio=8)
        self.ca_H1 = ChannelAttention(input_nc=64, ratio=8)
        self.ca_H2 = ChannelAttention(input_nc=128, ratio=8)


        # self.BasicBlockL1 = BasicBlock(128, 128)
        self.BasicBlockL2 = BasicBlock(256, 128)
        self.BasicBlockH1 = BasicBlock(64, 64)
        self.BasicBlockHLd1 = BasicBlock(256, 128)
        # self.max_pool_HL1 = nn.MaxPool2d(kernel_size=2)#32
        self.BasicBlockHLd2 = BasicBlock(128, 64)
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv_classier = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.up_H = nn.PixelShuffle(upscale_factor=2)
        self.up_L = nn.PixelShuffle(upscale_factor=2)

    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        L = input_data[:, 3, :, :]
        L = L.unsqueeze(1)
        input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]

        edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        # weight_edge = inputData * (1 + edge)  # 高频
        weight_edge=inputData+edge
        # weight_edge=self.sigmoid1(weight_edge)
        blur = F.conv2d(inputData, self.blur_weight, stride=1, padding=1, groups=3)
        # blur=self.sigmoid2(blur)

        return weight_edge, blur

    def extractorH(self, input1,input2):
        fea1 = self.backboneH.conv1(input1)
        fea1 = self.backboneH.bn1(fea1)
        fea1 = self.backboneH.relu(fea1)
        featH1_0 = self.backboneH.maxpool(fea1)
        featH1_1 = self.backboneH.layer1(featH1_0)
        featH1_1=self.ca_H1(featH1_1)*featH1_1
        refeatH1 = self.backboneH.layer2(featH1_1)
        refeatH1=self.ca_H2(refeatH1)*refeatH1

        fea2 = self.backboneH.conv1(input2)
        fea2 = self.backboneH.bn1(fea2)
        fea2 = self.backboneH.relu(fea2)
        featH2_0 = self.backboneH.maxpool(fea2)
        # featH2_0 = self.backboneH.layer0(input2)
        featH2_1 = self.backboneH.layer1(featH2_0)
        featH2_1=self.ca_H1(featH2_1)*featH2_1
        refeatH2 = self.backboneH.layer2(featH2_1)
        refeatH2=self.ca_H2(refeatH2)*refeatH2

        return featH1_0,featH1_0,featH1_1,featH2_1,refeatH1,refeatH2
    def extractorL(self, input1,input2):
        fea1 = self.backboneL.conv1(input1)
        fea1 = self.backboneL.bn1(fea1)
        fea1 = self.backboneL.relu(fea1)
        refeatL1_0 = self.backboneL.maxpool(fea1)
        # refeatL1_0 = self.backboneL.layer0(input1)
        refeatL1_1 = self.backboneL.layer1(refeatL1_0)
        refeatL1_1=self.ca_L1(refeatL1_1)*refeatL1_1
        refeatL1 = self.backboneL.layer2(refeatL1_1)
        refeatL1 = self.ca_L2(refeatL1)*refeatL1
        fea2 = self.backboneL.conv1(input2)
        fea2 = self.backboneL.bn1(fea2)
        fea2 = self.backboneL.relu(fea2)
        refeatL2_0 = self.backboneL.maxpool(fea2)
        # refeatL2_0 = self.backboneL.layer0(input2)
        refeatL2_1 = self.backboneL.layer1(refeatL2_0)
        refeatL2_1=self.ca_L1(refeatL2_1)*refeatL2_1
        refeatL2 = self.backboneL.layer2(refeatL2_1)
        refeatL2=self.ca_L2(refeatL2)*refeatL2
        # print('refeatL2',refeatL2.shape)
        featLcat = torch.cat([refeatL1, refeatL2], dim=1)#torch.Size([2, 256, 32, 32])
        # featLcat1=self.BasicBlockL1(featLcat)#128
        featLcat2 = self.BasicBlockL2(featLcat)#torch.Size([2, 128, 32, 32])
        # print('featLcat2', featLcat2.shape)
        return refeatL1_0,refeatL2_0,refeatL1_1,refeatL2_1,refeatL1,refeatL2,featLcat2
    def HLencoderD(self,HL1,HL2):
        featHL1_1 = self.BasicBlockHLd1(HL1)
        featHL2_1 = self.BasicBlockHLd1(HL2)#64*32*32

        featHL1_2= self.BasicBlockHLd2(featHL1_1)#64*32*32
        featHL2_2 = self.BasicBlockHLd2(featHL2_1)
        return featHL1_1,featHL2_1,featHL1_2,featHL2_2


    def forward(self, pre_data, post_data):
        # pre_dataLog=torch.log(pre_data+1)
        # post_dataLog=torch.log(post_data+1)
        inputH1, inputL1 = self.filterHL(pre_data)
        inputH2, inputL2 = self.filterHL(post_data)

        featH1_0,featH2_0,featH1_1,featH2_1,refeatH1,refeatH2 = self.extractorH(inputH1,inputH2)#torch.Size([4, 64, 64, 64]) torch.Size([4, 128, 32, 32])
        featL1_0,featL2_0,featL1_1,featL2_1,refeatL1,refeatL2,refeatL = self.extractorL(inputL1,inputL2)#torch.Size([4, 64, 64, 64])  torch.Size([4, 128, 32, 32])
        # featHL1=torch.cat([refeatH1,refeatL],dim=1)#128
        # featHL2 = torch.cat([refeatH2, refeatL], dim=1)#torch.Size([4, 128, 64, 64])
        # featHL1 = refeatH1 + refeatL#torch.Size([4, 128, 64, 64])
        # featHL2 = refeatH2 + refeatL
        featHL1=torch.cat([refeatH1,refeatL],dim=1)
        featHL2 = torch.cat([refeatH2, refeatL], dim=1)
        # print('featHL1', featHL1.shape)
        featHLDown1_l1,featHLDown2_l2,featHLDown1,featHLDown2=self.HLencoderD(featHL1,featHL2)#torch.Size([4, 64, 32, 32])
        # print('featHLDown1_l1', featHLDown1_l1.requires_grad)
        # print('featHLDown1', featHLDown1.shape)
        # featHLDown1Exp = torch.exp(featHLDown1)
        # featHLDown2Exp = torch.exp(featHLDown2)
        # featcat=torch.cat([featHLDown1Exp,featHLDown2Exp,torch.abs(featHLDown1Exp-featHLDown2Exp)],dim=1)#torch.Size([4, 192, 32, 32])
        featcat=torch.cat([featHLDown1,featHLDown2,torch.abs(featHLDown1-featHLDown2)],dim=1)#torch.Size([4, 192, 32, 32])

        HLup1 = self.deconv_1(featcat)#torch.Size([4, 64, 64, 64])
        # print('HLup1',HLup1.shape)
        # HLup1_cat=torch.cat([torch.abs(torch.exp(featH1_1)-torch.exp(featH2_1)),
        #                      torch.abs(torch.exp(featL1_1)-torch.exp(featL2_1)),HLup1],dim=1)#torch.Size([4, 192, 64, 64])
        HLup1_cat = torch.cat([torch.abs(featH1_1 - featH2_1),
                               torch.abs(featL1_1 - featL2_1), HLup1],dim=1)  # torch.Size([4, 192, 64, 64])
        # print('HLup1_cat', HLup1_cat.shape)
        HLup2=self.deconv_2(HLup1_cat)#torch.Size([4, 64, 128, 128])
        # print('HLup2', HLup2.shape)
        diffL_0 = self.up_L(torch.abs(featL1_0 - featL2_0))#
        diffH_0 = self.up_H(torch.abs(featH1_0 - featH2_0))#torch.Size([4, 16, 128, 128])
        # print('diffL_0', diffL_0.shape)
        HLup2_cat = torch.cat([diffL_0,diffH_0, HLup2], dim=1)  # torch.Size([4, 96, 128, 128])
        pre=self.deconv_classier(HLup2_cat)
        # self.encoder(post_data)
        return pre,pre

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