
import torch
import torch.nn as nn
import torch.nn.functional as F
class ASSP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ASSP,self).__init__()
        dilations=[1,6,12,18]

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0],#cannot set kernalsize to 3,otherwise cannot converge
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
        # residual = self.relu(x)
        residual = self.convplus(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class UCDNetRes(nn.Module):
    def __init__(self, in_dim=3,out_dim=2):
        super(UCDNetRes, self).__init__()
        # hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),

            nn.ReLU()
        )
        # self.conv_1_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #
        #     nn.ReLU()
        # )
        self.resconv_1_2=BasicBlock(16,16)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        #stage2
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),

            nn.ReLU()
        )
        # self.conv_2_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #
        #     nn.ReLU()
        # )
        self.resconv_2_2 = BasicBlock(32, 32)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        #stage3
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU()
        )
        # self.conv_3_2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        self.resconv_3_2 = BasicBlock(64, 64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        #stage4
        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU()
        )
        # self.conv_4_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #
        #     nn.ReLU()
        # )
        self.resconv_4_2 = BasicBlock(128, 128)
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv_diff_1= nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),

            nn.ReLU()
        )
        self.conv_diff_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),

            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),

            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),

            nn.ReLU()
        )
        # self.NSSP=NSSP(192,192,32)
        # self.ASSP=ASSP(192,192)
        # self.ASSPconv = nn.Sequential(
        #     nn.Conv2d(192, 192, kernel_size=1, stride=1, bias=False),
        #     # nn.BatchNorm2d(192),
        #     nn.ReLU()
        # )
        # self.resconvASSP = BasicBlock(192, 192)#add relu!!!!!!!!
        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),

            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        pre_data = pre_data[:, 0:3, :, :]
        post_data = post_data[:, 0:3, :, :]
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        # feature_T1_12 = self.conv_1_2(feature_T1_11)#relu-conv-relu
        feature_T1_12=self.resconv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        # feature_T2_12 = self.conv_1_2(feature_T2_11)
        feature_T2_12 = self.resconv_1_2(feature_T2_11)

        diff1=torch.abs(feature_T1_11-feature_T2_11)
        diff1=self.conv_diff_1(diff1)#relu-conv-relu

        feature_T1_13=torch.cat([feature_T1_12,diff1],1)
        feature_T1_14=self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21=self.conv_2_1(feature_T1_14)#conv
        # feature_T1_22=self.conv_2_2(feature_T1_21)#relu-conv-relu
        feature_T1_22 = self.resconv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        # feature_T2_22 = self.conv_2_2(feature_T2_21)
        feature_T2_22 = self.resconv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)

        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        # feature_T1_32 = self.conv_3_2(feature_T1_31)#relu-conv-relu
        feature_T1_32 = self.resconv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        # feature_T2_32 = self.conv_3_2(feature_T2_31)
        feature_T2_32 = self.resconv_3_2(feature_T2_31)#relu-conv-relu



        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3], 1)#128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        # feature_T1_42 = self.conv_4_2(feature_T1_41)#relu-conv-relu
        feature_T1_42 = self.resconv_4_2(feature_T1_41)  # relu-conv-relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        # feature_T2_42 = self.conv_4_2(feature_T2_41)
        feature_T2_42 = self.resconv_4_2(feature_T2_41)  # relu-conv-relu

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4], 1)
        feature_T1_44=self.conv_4_3(feature_T1_43)#conv
        feature_T2_43 = torch.cat([feature_T2_42, diff4], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)

        diff5=torch.abs(feature_T1_44-feature_T2_44)

        feature_Bottleneck=torch.cat([feature_T1_44,feature_T2_44,diff5],1)
        # feature_Bottleneck = self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)
        # feature_Bottleneck = self.resconvASSP(feature_Bottleneck)

        decode_1=self.up_sample_1(feature_Bottleneck)
        decode_1=torch.cat([feature_T1_33,feature_T2_33,decode_1],1)#320

        decode_2=self.deconv_1(decode_1)
        decode_2=torch.cat([feature_T1_23,feature_T2_23,decode_2],1)

        decode_3=self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13,feature_T2_13,decode_3],1)

        outfeature=self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature, outfeature
class UCDNet_ASSPMultiout(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(UCDNet_ASSPMultiout, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff1_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU()
        )


        self.conv_diff_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff2_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32 , kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32 , 32, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0)
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff3_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.conv_diff_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_diff_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.BatchNorm2d(64),
            # nn.ReLU()
            # nn.Sigmoid()
        )
        self.diff5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        # self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(64, 64)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )
        self.updiff3=nn.Upsample(scale_factor=4, mode='bilinear')
        self.updiff4=nn.Upsample(scale_factor=8, mode='bilinear')
        self.updiff5=nn.Upsample(scale_factor=8, mode='bilinear')

        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.softmax = nn.Softmax()
        self.softsign=nn.Softsign()
        self.tanh=nn.Tanh()

    def forward(self, pre_data,post_data):
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)
        feature_T1_12 = self.conv_1_2(feature_T1_11)
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)
        diff1=diff1*self.diff1_se(diff1)
        diff1_weight = self.relu(torch.abs(diff1))

        # diff1_weight = diff1
        feature_T1_13 = torch.cat([feature_T1_12, diff1_weight], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1_weight], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        feature_T1_21 = self.conv_2_1(feature_T1_14)
        feature_T1_22 = self.conv_2_2(feature_T1_21)
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        diff2 = diff2 * self.diff2_se(diff2)
        diff2_weight=self.relu(torch.abs(diff2))
        # diff2_weight=diff2
        feature_T1_23 = torch.cat([feature_T1_22, diff2_weight], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2_weight], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        feature_T1_31 = self.conv_3_1(feature_T1_24)
        feature_T1_32 = self.conv_3_2(feature_T1_31)
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        diff3 = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(torch.abs(diff3))
        # diff3_weight=diff3
        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)

        feature_T1_41 = self.conv_4_1(feature_T1_34)
        feature_T1_42 = self.conv_4_2(feature_T1_41)
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        diff4 = diff4 * self.diff4_se(diff4)
        # diff4_weight=self.tanh(diff4)
        diff4_weight = self.relu(torch.abs(diff4))

        # diff4_weight=diff4
        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)
        feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_45 - feature_T2_45)  # 64
        diff5 = self.conv_diff_5(diff5)

        diff5 = diff5 * self.diff5_se(diff5)
        # diff5_weight=self.relu(diff5)
        diff5_weight = self.relu(torch.abs(diff5))
        # diff5_weight=diff5
        # print('diff5_weight',diff5_weight.shape)
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)

        # print('feature_Bottleneck', feature_Bottleneck.shape)
        # feature_Bottleneck=self.ASSP(feature_Bottleneck)

        decode_1 = self.up_sample_1(feature_Bottleneck)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        decode_3 = self.deconv_2(decode_2)#150
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, decode_3], 1)#80

        outfeature = self.deconv_3(decode_3)
        # diff3_weight=self.updiff3(diff3_weight)
        # diff4_weight = self.updiff4(diff4_weight)
        # diff5_weight = self.updiff5(diff5_weight)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        # return [outfeature,diff4_weight,diff5_weight], decode_3

        return outfeature,outfeature
class UCDNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(UCDNet, self).__init__()
        hideen_num = [16, 32, 64, 128]
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # nn.ReLU()
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.ReLU()
        )
        self.conv_2_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU()
        )
        self.conv_3_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU()
        )
        self.conv_4_2 = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_4_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU()
        )

        self.conv_diff_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # self.diff1_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, kernel_size=1),
        #     nn.Sigmoid()
        # )


        self.conv_diff_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.diff2_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(32, 32 , kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32 , 32, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.diff3_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.diff4_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.conv_diff_5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.diff5_se = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # )
        # self.NSSP = NSSP(192, 192, 32)
        self.ASSP = ASSP(192, 192)
        self.ASSPconv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu=nn.ReLU()
    def forward(self, pre_data,post_data):
        pre_data = pre_data[:, 0:3, :, :]
        post_data = post_data[:, 0:3, :, :]
        #####################
        # decoder
        #####################
        # post_data=pre_data
        feature_T1_11 = self.conv_1_1(pre_data)#conv
        feature_T1_12 = self.conv_1_2(feature_T1_11)#relu+conv+relu
        feature_T2_11 = self.conv_1_1(post_data)
        feature_T2_12 = self.conv_1_2(feature_T2_11)

        diff1 = torch.abs(feature_T1_11 - feature_T2_11)
        diff1 = self.conv_diff_1(diff1)
        # diff1_weight=diff1*self.diff1_se(diff1)
        # diff1_weight=self.relu(diff1)
        feature_T1_13 = torch.cat([feature_T1_12, diff1], 1)
        feature_T1_14 = self.max_pool_1(feature_T1_13)
        feature_T2_13 = torch.cat([feature_T2_12, diff1], 1)
        feature_T2_14 = self.max_pool_1(feature_T2_13)

        #Stage 2

        feature_T1_21 = self.conv_2_1(feature_T1_14)#conv
        feature_T1_22 = self.conv_2_2(feature_T1_21)#relu+conv+relu
        feature_T2_21 = self.conv_2_1(feature_T2_14)
        feature_T2_22 = self.conv_2_2(feature_T2_21)

        diff2 = torch.abs(feature_T1_21 - feature_T2_21)
        diff2 = self.conv_diff_2(diff2)
        # diff2_weight = diff2 * self.diff2_se(diff2)
        # diff2_weight = self.relu(diff2)
        feature_T1_23 = torch.cat([feature_T1_22, diff2], 1)
        feature_T1_24 = self.max_pool_2(feature_T1_23)
        feature_T2_23 = torch.cat([feature_T2_22, diff2], 1)
        feature_T2_24 = self.max_pool_2(feature_T2_23)

        #Stage 3

        feature_T1_31 = self.conv_3_1(feature_T1_24)#conv
        feature_T1_32 = self.conv_3_2(feature_T1_31)#relu+conv+relu
        feature_T2_31 = self.conv_3_1(feature_T2_24)
        feature_T2_32 = self.conv_3_2(feature_T2_31)

        diff3 = torch.abs(feature_T1_31 - feature_T2_31)
        diff3 = self.conv_diff_3(diff3)
        # diff3_weight = diff3 * self.diff3_se(diff3)
        diff3_weight = self.relu(diff3)

        feature_T1_33 = torch.cat([feature_T1_32, diff3_weight], 1)
        feature_T1_34 = self.max_pool_3(feature_T1_33)
        feature_T2_33 = torch.cat([feature_T2_32, diff3_weight], 1)  # 128
        feature_T2_34 = self.max_pool_3(feature_T2_33)
        #stage 4
        feature_T1_41 = self.conv_4_1(feature_T1_34)#conv
        feature_T1_42 = self.conv_4_2(feature_T1_41)#relu+conv+relu
        feature_T2_41 = self.conv_4_1(feature_T2_34)
        feature_T2_42 = self.conv_4_2(feature_T2_41)

        diff4 = torch.abs(feature_T1_41 - feature_T2_41)
        diff4 = self.conv_diff_4(diff4)
        # diff4_weight = diff4 * self.diff4_se(diff4)
        diff4_weight = self.relu(diff4)

        feature_T1_43 = torch.cat([feature_T1_42, diff4_weight], 1)
        feature_T1_44 = self.conv_4_3(feature_T1_43)#conv
        # feature_T1_45 = self.ASSP(feature_T1_44)
        # feature_T1_45 = self.ASSPconv(feature_T1_44)
        # feature_T1_45=torch.cat([feature_T1_44,feature_T1_45],1)
        feature_T2_43 = torch.cat([feature_T2_42, diff4_weight], 1)
        feature_T2_44 = self.conv_4_3(feature_T2_43)
        # feature_T2_45 = self.ASSP(feature_T2_44)
        # feature_T2_45 = self.ASSPconv(feature_T2_44)

        # feature_T2_45 = torch.cat([feature_T2_44, feature_T2_45], 1)

        diff5 = torch.abs(feature_T1_44 - feature_T2_44)  # 64
        # diff5 = self.conv_diff_5(diff5)
        # diff5_weight = diff5 * self.diff5_se(diff5)
        diff5_weight=diff5
        feature_Bottleneck = torch.cat([feature_T1_44, feature_T2_44, diff5_weight], 1)
        feature_Bottleneck=self.relu(feature_Bottleneck)
        # print('feature_Bottleneck', feature_Bottleneck.shape)
        feature_BottleneckASSP=self.ASSP(feature_Bottleneck)
        ASSPconv_bottle = self.ASSPconv(feature_Bottleneck)
        feature_Bottleneckout=torch.cat([feature_BottleneckASSP,ASSPconv_bottle],1)
        decode_1 = self.up_sample_1(feature_Bottleneckout)
        decode_1 = torch.cat([feature_T1_33, feature_T2_33, decode_1], 1)  # 320

        decode_2 = self.deconv_1(decode_1)
        decode_2 = torch.cat([feature_T1_23, feature_T2_23, decode_2], 1)

        DA = self.deconv_2(decode_2)
        decode_3 = torch.cat([feature_T1_13, feature_T2_13, DA], 1)

        outfeature = self.deconv_3(decode_3)
        # outfeature=self.sigmoid(outfeature)
        # outfeature = self.softmax(outfeature)
        # print('outfeature',outfeature.shape)
        return outfeature,DA
    def freeze_bn_dr(self):
        # if a==True:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.Dropout):
                module.eval()