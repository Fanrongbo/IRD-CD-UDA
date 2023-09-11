import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
import numpy as np
from torchvision import models
# from models.appendix import ASSP,BaseModel,initialize_weights
def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))
class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride,mid_channel=256):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, mid_channel, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, mid_channel, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, mid_channel, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, mid_channel, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, mid_channel, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU())

        self.conv1 = nn.Conv2d(mid_channel * 5, mid_channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


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
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet34', pretrained=False,AG_flag=False):
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
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        initialize_weights(self.layer3)
        initialize_weights(self.layer4)

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
        # self.diff_ag2 = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.diff_ag3 = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=1),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.diff_ag4 = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.Conv2d(256, 128, kernel_size=1),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.diff_se2 = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64 // 2, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.diff_se3 = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128 // 2, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.diff_se4 = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(256, 256 // 2, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256 // 2, 256, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.AG_flag=AG_flag
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
    def forward(self, x1, x2):
        x11 = self.layer0(x1)
        x12 = self.layer0(x2)
        ########stage 2
        x21 = self.layer1(x11)
        x22 = self.layer1(x12)
        diff2 = torch.abs(x21 - x22)
        x21 = x21 + diff2
        x22 = x22 + diff2
        ########stage 3
        x31 = self.layer2(x21)
        x32 = self.layer2(x22)
        diff3 = torch.abs(x31 - x32)

        if self.AG_flag:
            # diff3_AG = self.diff_ag3(diff3)
            x31 = x31 + diff3
            x32 = x32 + diff3
        else:
            x31 = x31 + diff3
            x32 = x32 + diff3
        ########stage 4
        x41 = self.layer3(x31)
        x42 = self.layer3(x32)
        diff4 = torch.abs(x41 - x42)
        if self.AG_flag:
            x41 = x41 +diff4
            x42 = x42 +diff4
        else:
            x41 = x41+diff4
            x42 = x42+diff4

        if self.AG_flag:
            return [x11, x21, x31, x41], [x12, x22, x32, x42],[diff2,diff3,diff4]
        else:
            return [x11, x21, x31, x41], [x12, x22, x32, x42]

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes,asspflag=True):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        if asspflag:
            # Table 2, best performance with two 3x3 convs
            self.output = nn.Sequential(
                nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1, stride=1),
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(48 + low_level_channels, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1, stride=1),
            )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x
class DeepLab(BaseModel):
    def __init__(self, in_channels=3,num_classes=2,  backbone='resnet', pretrained=False,
                 output_stride=16, freeze_bn=False, **_):

        super(DeepLab, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=False,AG_flag=False)
            low_level_channels = 64
        # else:
        #     self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
        #     low_level_channels = 128

        self.ASSP = ASSP(in_channels=512, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()

    def forward(self, x1,x2):
        # x2=x1
        x1 = x1[:, 0:3, :, :]
        x2 = x2[:, 0:3, :, :]
        feature1,feature2=self.backbone(x1,x2)
        H, W = x1.size(2), x1.size(3)
        x=torch.cat([feature1[-1],feature2[-1]],dim=1)
        low_level_diff=torch.abs(feature1[1]-feature2[1])
        # low_level_features=torch.cat([feature1[1],feature2[1]],dim=1)
        x = self.ASSP(x)

        x_DA = self.decoder(x, low_level_diff)
        x = F.interpolate(x_DA, size=(H, W), mode='bilinear', align_corners=True)
        # diffout = F.interpolate(diffout, size=(H, W), mode='bilinear', align_corners=True)

        return x, x_DA
    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


