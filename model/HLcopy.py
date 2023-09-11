from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
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


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet34', pretrained=True):
        super(ResNet, self).__init__()
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
        # self.layer3 = model.layer3
        # self.layer4 = model.layer4
        # initialize_weights(self.layer3)
        # initialize_weights(self.layer4)

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

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
        x11 = self.layer0(x1)
        # x12 = self.layer0(x2)

        x21 = self.layer1(x11)
        # x22 = self.layer1(x12)
        # diff2 = torch.abs(x21 - x22)
        # x31 = self.layer2(x21)
        # x32 = self.layer2(x22)
        #
        # x41 = self.layer3(x31)
        # x42 = self.layer3(x32)

        return x11, x21


class HLCDNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(HLCDNet, self).__init__()
        # sobel_kernel = np.array([[-1, -1, -1],	[-1, 8, -1],	[-1, -1, -1]],	dtype='float32')
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))))
        # self.convsobelX = nn.Conv2d(1, 1, 3,stride=1,padding=1, bias=False)
        # self.convsobelX.weight.data = torch.from_numpy(sobel_kernel_x)
        # self.convsobelX.weight.requires_grad=False
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))))
        # self.convsobelY = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        # self.convsobelY.weight.data = torch.from_numpy(sobel_kernel_y)
        # self.convsobelY.weight.requires_grad = False
        gussiankernal = self.get_gaussian_kernel2()
        self.blur_weight = nn.Parameter(torch.from_numpy(gussiankernal))
        # gussiankernal = gussiankernal.repeat(3, 1, 1, 1)
        # self.blur_layer = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        # self.blur_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,
        #                              bias=False, padding=1)
        # self.blur_layer.weight.data = (torch.from_numpy(gussiankernal)).repeat(3,1,1,1)
        # self.blur_layer.weight.requires_grad = False
        # self.blur_layer = self.get_gaussian_kernel()
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.backboneH = ResNet(in_channels=3, output_stride=2, pretrained=True)
        self.backboneL = ResNet(in_channels=3, output_stride=2, pretrained=True)

    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        L = input_data[:, 3, :, :]
        L = L.unsqueeze(1)
        # input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]
        # edgeX=self.convsobelX(L)
        # edgeY = self.convsobelX(L)
        edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        weight_edge = inputData * (1 + edge)  # 高频
        # blur = self.blur_layer(inputData)  # 低频
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
        return weight_edge, blur

    def extractorH(self, input):
        _,refeat=self.backboneH(input)#torch.Size([2, 64, 64, 64])
        return refeat
    def extractorL(self, input1,input2):
        _,refeatL1=self.backboneL(input1)#torch.Size([2, 64, 64, 64])
        _, refeatL2 = self.backboneL(input2)  # torch.Size([2, 64, 64, 64])
        featL = torch.cat([refeatL1, refeatL2], dim=1)

        return refeat
    def forward(self, pre_data, post_data):
        # pre_dataLog=torch.log(pre_data)
        # post_dataLog=torch.log(post_data)
        inputH1, inputL1 = self.filterHL(pre_data)
        inputH2, inputL2 = self.filterHL(post_data)
        refeatH1=self.extractorH(inputH1)
        refeatH2 = self.extractorH(inputH2)
        refeatL1 = self.extractorL(inputL1)
        refeatL2 = self.extractorL(inputL2)


        # self.encoder(post_data)
        # return pre_data,post_data

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
