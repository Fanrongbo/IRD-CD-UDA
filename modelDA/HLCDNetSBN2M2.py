from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modelDA.resnet import resnet34_2l
import matplotlib.pyplot as plt
from modelDA.domain_specific_module import BatchNormDomain
class ASSP(nn.Module):
    def __init__(self,in_channels,out_channels, num_domains_bn=2):
        super(ASSP,self).__init__()
        dilations=[1,6,12,18]
        self.num_domains_bn = num_domains_bn
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0],#cannot set kernalsize to 3,otherwise cannot converge
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            BatchNormDomain(out_channels, self.num_domains_bn, nn.BatchNorm2d),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            BatchNormDomain(out_channels, self.num_domains_bn, nn.BatchNorm2d),

            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            BatchNormDomain(out_channels, self.num_domains_bn, nn.BatchNorm2d),

            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            # nn.BatchNorm2d(out_channels),
            BatchNormDomain(out_channels, self.num_domains_bn, nn.BatchNorm2d),

            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            # nn.BatchNorm2d(out_channels),
            BatchNormDomain(out_channels, self.num_domains_bn, nn.BatchNorm2d),

            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            BatchNormDomain(out_channels, self.num_domains_bn, nn.BatchNorm2d),
            nn.ReLU()
            # nn.ReLU6()
        )
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        x5 = self.layer5(x)
        # print(x5.shape)
        x5=F.upsample(x5,size=x1.size()[2:],mode='bilinear',align_corners=True)
        x=torch.cat((x1,x2,x3,x4,x5),dim=1)
        # print('x',x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,)
        x=self.layer6(x)

        return x
    def set_bn_domain(self, domain=0):
        self.domain = domain
        for m in self.modules():  # get the name from all of the moudule
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)
class ChannelAttention(nn.Module):
    def __init__(self, input_nc, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(input_nc, input_nc // ratio, 1, bias=True)
        # self.bn1 = nn.BatchNorm2d(input_nc // ratio)
        self.relu1 = nn.ReLU(inplace=True)
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
    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None,num_domains_bn=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = BatchNormDomain(planes, num_domains_bn, nn.BatchNorm2d)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = BatchNormDomain(planes, num_domains_bn, nn.BatchNorm2d)
        self.downsample = downsample
        self.stride = stride
        self.convplus = nn.Conv2d(inplanes, planes, stride=stride,kernel_size=1, bias=False)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

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
    def __init__(self, in_channels=192, out_channels=128, kernel_size=3, padding=1,stride=1,num_domains_bn=0):
        super(basicConv, self).__init__()
        if stride==1:
            self.conv=nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                BatchNormDomain(out_channels, num_domains_bn, nn.BatchNorm2d),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif stride>1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
                BatchNormDomain(out_channels, num_domains_bn, nn.BatchNorm2d),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)
    def forward(self,x):
        out=self.conv(x)
        return out

class BaseNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=2,pretrained=True,num_domains_bn=2):
        super(BaseNet, self).__init__()
        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn
        # print('self.num_domains_bn',self.num_domains_bn)

        # self.Lmoudle= nn.ModuleDict()#module 的 parameters 添加到网络之中的容器


        self.backboneH = resnet34_2l(num_domains=self.num_domains_bn, pretrained=pretrained,
                                     replace_stride_with_dilation=[False, False, True])  # resnet34
        self.BasicBlockHLd1 = nn.Sequential(
            BasicBlock(128, 128, num_domains_bn=self.num_domains_bn),
            # BasicBlock(128, 128)
        )

    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

    def set_bn_domain(self, domain=0):
        self.domain = domain
        for m in self.modules():  # get the name from all of the moudule
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

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
        return fea1, fea2, featH1_1, featH2_1, featH1_2, featH2_2


    def forward(self, inputH1, inputH2,L_AC):


        featH1_0, featH2_0, featH1_1, featH2_1, featH1_2, featH2_2 = self.extractorH(inputH1,
                                                                                     inputH2)  # torch.Size([60, 64, 64, 64]) #[60, 128, 32, 32]
        CAW1 = featH1_2 * (1 + L_AC)
        CAW2 = featH2_2 * (1 + L_AC)  # [60, 128, 32, 32]
        Fusion1_1 = self.BasicBlockHLd1(CAW1)  # ([60, 128, 32, 32])
        Fusion1_2 = self.BasicBlockHLd1(CAW2)  # ([60, 128, 32, 32])
        FusionFeat = torch.cat([Fusion1_1, Fusion1_2], dim=1)  # ([60, 256, 32, 32])
        featH=[featH1_0, featH2_0, featH1_1, featH2_1]
        return FusionFeat,featH
class LowEx(nn.Module):
    def __init__(self, num_domains_bn=2):
        super(LowEx, self).__init__()
        self.num_domains_bn = num_domains_bn
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
        self.sobel_kernel_x = nn.Parameter(torch.from_numpy(sobel_kernel_x.reshape((1, 1, 3, 3))),
                                           requires_grad=False)
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')
        self.sobel_kernel_y = nn.Parameter(torch.from_numpy(sobel_kernel_y.reshape((1, 1, 3, 3))),
                                           requires_grad=False)
        self.Lmoudle = nn.Sequential(
            # nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            basicConv(in_channels=3, out_channels=16, kernel_size=7, padding=3, num_domains_bn=self.num_domains_bn),
            # basicConv(in_channels=16, out_channels=16, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 128
            basicConv(in_channels=16, out_channels=32, kernel_size=7, padding=3, num_domains_bn=self.num_domains_bn),
            # basicConv(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64
            basicConv(in_channels=32, out_channels=64, kernel_size=3, padding=1, num_domains_bn=self.num_domains_bn),
            # basicConv(in_channels=64, out_channels=64, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32
            basicConv(in_channels=64, out_channels=128, kernel_size=3, padding=1, num_domains_bn=self.num_domains_bn),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  #
            # basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            SpatialAttention(kernel_size=7)
            # ChannelAttention(input_nc=128, ratio=8)

        )

    def extractorL(self, input1, input2):
        L_AC1 = self.Lmoudle(input1)
        L_AC2 = self.Lmoudle(input2)  # [60, 128, 1, 1]

        return L_AC1, L_AC2

    def filterHL(self, input_data):
        inputData = input_data[:, 0:3, :, :]
        L = input_data[:, 3, :, :]
        L = L.unsqueeze(1)
        # input_dataLog = torch.log(inputData+1)#[4, 3, 256, 256]

        edgeX = F.conv2d(L, self.sobel_kernel_x, stride=1, padding=1)
        edgeY = F.conv2d(L, self.sobel_kernel_y, stride=1, padding=1)
        # edge = torch.sqrt(edgeX ** 2 + edgeY ** 2)
        edge = edgeX ** 2 + edgeY ** 2
        # weight_edge = inputData * (1 + edge)  # 高频
        weight_edge = inputData + edge

        return weight_edge, inputData
    def forward(self, pre_data, post_data):
        inputH1, inputL1 = self.filterHL(pre_data)
        inputH2, inputL2 = self.filterHL(post_data)

        L_AC1, L_AC2 = self.extractorL(inputL1, inputL2)  # [60, 128, 1, 1]
        L_AC = (L_AC1 + L_AC2) / 2
        L_AC_out = torch.flatten(L_AC, start_dim=1, end_dim=3)
        return [inputH1,inputH2,L_AC],L_AC_out
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

    def set_bn_domain(self, domain=0):
        self.domain = domain
        for m in self.modules():  # get the name from all of the moudule
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)
class DeepDeconv(nn.Module):
    def __init__(self, num_domains_bn=2):
        super(DeepDeconv, self).__init__()
        self.num_domains_bn = num_domains_bn
        self.assp = ASSP(256, 256, self.num_domains_bn)
        self.deconv_1 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            # nn.BatchNorm2d(128),
            BatchNormDomain(128, self.num_domains_bn, nn.BatchNorm2d),
            nn.ReLU(inplace=True),

            BasicBlock(128, 128, num_domains_bn=self.num_domains_bn)
            # basicConv(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )
        self.deconv_2 = nn.Sequential(
            # basicConv(in_channels=128, out_channels=64, kernel_size=2, padding=1,stride=2),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            # nn.BatchNorm2d(64),
            BatchNormDomain(64, num_domains_bn, nn.BatchNorm2d),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64, num_domains_bn=self.num_domains_bn),
            # basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),

        )
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            # nn.BatchNorm2d(64),
            BatchNormDomain(64, num_domains_bn, nn.BatchNorm2d),
            nn.ReLU(inplace=True),
            # BasicBlock(64, 64),
            BasicBlock(64, 32, num_domains_bn=self.num_domains_bn),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # BatchNormDomain(32, num_domains_bn, nn.BatchNorm2d),
            # # nn.Tanh()
            # nn.ReLU6()
            # nn.ReLU()
            # basicConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # basicConv(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        )

    def forward(self,FusionFeat,featH):
        [featH1_0,featH2_0,featH1_1,featH2_1]=featH
        # print('FusionFeat',FusionFeat.shape)#[8, 256, 32, 32])
        Fassp=self.assp(FusionFeat)
        # print('Fassp',Fassp.shape)
        defeat1 = self.deconv_1(Fassp)  # ([60, 128, 64, 64]
        diffFeat1 = torch.abs(featH1_1 - featH2_1)
        defeat1_cat = torch.cat([defeat1, diffFeat1], dim=1)  # 60, 192, 64, 64
        defeat2 = self.deconv_2(defeat1_cat)
        diffFeat2 = torch.abs(featH1_0 - featH2_0)
        defeat2_cat = torch.cat([defeat2, diffFeat2], dim=1)  # [60, 128, 128, 128]
        defeat3 = self.deconv_3(defeat2_cat)  # 32, 32, 256, 256
        # defeat3=self.assp(defeat3)
        # defeat3=F.relu6(defeat3)
        return defeat3

    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()

    def set_bn_domain(self, domain=0):
        self.domain = domain
        for m in self.modules():  # get the name from all of the moudule
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

class CD_classifer(nn.Module):
    def __init__(self,num_domains_bn):
        super(CD_classifer, self).__init__()
        self.num_domains_bn = num_domains_bn
        self.deconv_classier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
        )
    def forward(self, defeat3):
        pre=self.deconv_classier(defeat3)
        return pre
    def set_bn_domain(self, domain=0):
        self.domain = domain
        for m in self.modules():  # get the name from all of the moudule
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)