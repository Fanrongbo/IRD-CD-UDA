import torch
import torch.nn as nn
import torch.nn.init as init
from option.config import cfg

from model.UCDNet import UCDNet,UCDNet_ASSPMultiout,UCDNetRes
from model.Siamese_diff import FCSiamDiff,FCSiamDiffBNnewMultiPSPBN,FCSiamConc
from model.deepLab import DeepLab
from model.FCN import FCN_ResNet
from model.ISNet import Model_FullMM,Model_PartialMM
from model.unet import Unet
from model.HL import HLCDNetlog,HLCDNet2
from model.HLnet import HLCDNetNo,HLCDNetG,HLCDNetGB
from model.HLnetS import HLCDNetS,HLCDNetS2,HLCDNetSL,HLCDNetSLS,HLCDNetSS,HLCDNetSSC
from modelDA.HLnetSDA import HLCDNetSBN
from model.DSIFN import DSIFN
from model.HLnetM0 import HLCDNetNoM0
# from model.Models import FCSiamConc


def init_method(net, init_type='normal'):
    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'resnet'):
            pass
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=0.02)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


#func called to initialize a net
def init_net(net, init_type='normal', initialize=True, gpu_ids=[]):
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net.to(DEVICE)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if initialize:
        init_method(net, init_type)
    else:
        pass
    return net


#print #parameters
def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



def define_model(model_type='PartialMM', resnet='resnet18', init_type='normal', initialize=True, gpu_ids=[]):
    net = network_dict[model_type]()
    # if model_type == 'FullMM':
    #     net = Model_FullMM(resnet=resnet)
    # elif model_type == 'PartialMM':
    #     net = Model_PartialMM(resnet=resnet)
    # else:
    #     raise NotImplementedError
    print_network(net)

    return init_net(net, init_type, initialize, gpu_ids)

class standard_unit(nn.Module):
    def __init__(self,in_filter,nb_filter,mode='residual'):
        # print('in_filter,nb_filter',in_filter,nb_filter)
        super(standard_unit, self).__init__()
        self.nb_filter=nb_filter
        self.in_filter=in_filter
        # self.kernel_size=kernel_size
        self.mode=mode
        self.conv1 = nn.Conv2d(self.in_filter, self.nb_filter, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.nb_filter)
        self.conv2 = nn.Conv2d(self.nb_filter, self.nb_filter, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nb_filter)
    def forward(self, x):
        x=self.conv1(x)
        x0=x
        x=self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.mode == 'residual':
            # x=torch.cat([x,x0],1)
            x=x+x0
        return x


class Nest_Net(nn.Module):
    def __init__(self, input_nbr=6, label_nbr=2):
        super(Nest_Net, self).__init__()
        self.input_nbr=input_nbr
        self.label_nbr=label_nbr
        nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter =nb_filter
        # self.standard_unit1=standard_unit()
        self.standard_unit_in_0=standard_unit(input_nbr,nb_filter[0],mode='none')
        self.standard_unit_0_1=standard_unit(nb_filter[0],nb_filter[1])
        self.standard_unit_1_2 = standard_unit(nb_filter[1], nb_filter[2])
        self.standard_unit_2_3 = standard_unit(nb_filter[2], nb_filter[3])
        self.standard_unit_3_4 = standard_unit(nb_filter[3], nb_filter[4])
        self.standard_unit_0_3 = standard_unit(nb_filter[0], nb_filter[3])
        self.standard_unit_0_4 = standard_unit(nb_filter[0], nb_filter[4])

        self.ConvTranspose2d_1_0 = nn.ConvTranspose2d(nb_filter[1],nb_filter[0], kernel_size=2, stride=2)
        self.ConvTranspose2d_1_0_2 = nn.ConvTranspose2d(nb_filter[1],nb_filter[0], kernel_size=2, stride=2)
        self.ConvTranspose2d_1_0_3 = nn.ConvTranspose2d(nb_filter[1],nb_filter[0], kernel_size=2, stride=2)
        self.ConvTranspose2d_1_0_4 = nn.ConvTranspose2d(nb_filter[1],nb_filter[0], kernel_size=2, stride=2)

        self.ConvTranspose2d_2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.ConvTranspose2d_2_1_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.ConvTranspose2d_2_1_3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)

        self.ConvTranspose2d_3_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.ConvTranspose2d_3_2_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.ConvTranspose2d_3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[1], kernel_size=2, stride=2)
        self.ConvTranspose2d_4_3 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)

        self.standard_unit_1_0= standard_unit(nb_filter[1], nb_filter[0])
        self.standard_unit_2_1 = standard_unit(nb_filter[2], nb_filter[1])
        self.standard_unit_3_1=standard_unit(nb_filter[3], nb_filter[1])
        self.standard_unit_3_2 = standard_unit(nb_filter[3], nb_filter[2])
        self.standard_unit_4_3 = standard_unit(nb_filter[4], nb_filter[3])
        self.standard_unit_0_2 = standard_unit(nb_filter[0], nb_filter[2])
        self.standard_unit_96_0 = standard_unit(nb_filter[0]*3, nb_filter[0])
        self.standard_unit_192_1 = standard_unit(nb_filter[1] * 3, nb_filter[1])
        self.standard_unit_128_0 = standard_unit(nb_filter[0] * 4, nb_filter[0])
        self.standard_unit_384_2 = standard_unit(nb_filter[2] * 3, nb_filter[2])
        self.standard_unit_256_1 = standard_unit(nb_filter[1] * 4, nb_filter[1])
        self.standard_unit_160_0 = standard_unit(nb_filter[0] * 5, nb_filter[0])

        self.conv = nn.Conv2d(nb_filter[0], label_nbr, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nb_filter[0], label_nbr, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(nb_filter[0], label_nbr, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nb_filter[0], label_nbr, kernel_size=3, padding=1, bias=False)


        self.conv5 = nn.Conv2d(nb_filter[0]*4, label_nbr, kernel_size=3, padding=1, bias=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x1,x2):
        input=torch.cat([x1[:,:3,:,:],x2[:,:3,:,:]],1)
        # input=x
        # print('Nest_Net2')
        conv1_1=self.standard_unit_in_0(input)#32
        pool1=self.pool(conv1_1)#32

        conv2_1=self.standard_unit_0_1(pool1)#64(None, 128, 128, 64)
        pool2 = self.pool(conv2_1)#64

        up1_2=self.ConvTranspose2d_1_0(conv2_1)#32 (None, 256, 256, 32)

        conv1_2=torch.cat([up1_2, conv1_1],1)#64 (None, 256, 256, 64)

        conv1_2=self.standard_unit_1_0(conv1_2)#32

        conv3_1=self.standard_unit_1_2(pool2)#128
        pool3=self.pool(conv3_1)# (None, 32, 32, 128)

        up2_2=self.ConvTranspose2d_2_1(conv3_1)#64
        conv2_2=torch.cat([up2_2, conv2_1],1)#128

        conv2_2=self.standard_unit_2_1(conv2_2)#64

        up1_3=self.ConvTranspose2d_1_0_2(conv2_2)#32
        conv1_3=torch.cat([up1_3, conv1_1, conv1_2],1)#96
        conv1_3=self.standard_unit_96_0(conv1_3)#32

        conv4_1=self.standard_unit_2_3(pool3)#256 # (?,32,32,256)
        pool4=self.pool(conv4_1)#256)# (?,64,64,256) (16,16,256)

        up3_2=self.ConvTranspose2d_3_2(conv4_1)#128#(?,64,64,128)
        conv3_2=torch.cat([up3_2, conv3_1],1)#256

        conv3_2=self.standard_unit_3_2(conv3_2)#128

        up2_3=self.ConvTranspose2d_2_1_2(conv3_2)#64
        conv2_3=torch.cat([up2_3, conv2_1, conv2_2],1)#192
        conv2_3=self.standard_unit_192_1(conv2_3)#64

        up1_4=self.ConvTranspose2d_1_0_3(conv2_3)
        conv1_4=torch.cat([up1_4, conv1_1, conv1_2, conv1_3],1)#(None, 256, 256, 128
        conv1_4=self.standard_unit_128_0(conv1_4)#(None, 256, 256, 32

        conv5_1 =  self.standard_unit_3_4(pool4)#(None, 16, 16, 512)

        up4_2=self.ConvTranspose2d_4_3(conv5_1)#(None, 32, 32, 256)
        conv4_2=torch.cat([up4_2, conv4_1],1)#(None, 32, 32, 512)

        conv4_2=self.standard_unit_4_3(conv4_2)#(None, 32, 32, 256)

        up3_3=self.ConvTranspose2d_3_2_2(conv4_2)#(None, 64, 64, 128)
        conv3_3=torch.cat([up3_3, conv3_1, conv3_2],1)##(None, 64, 64, 128*3)
        conv3_3=self.standard_unit_384_2(conv3_3)#(None, 64, 64, 128)

        up2_4=self.ConvTranspose2d_2_1_3(conv3_3)#(None, 128, 128, 64)
        conv2_4=torch.cat([up2_4, conv2_1, conv2_2, conv2_3],1)#(None, 128, 128, 64*4)
        conv2_4=self.standard_unit_256_1(conv2_4)#(None, 128, 128, 64)

        up1_5=self.ConvTranspose2d_1_0_4(conv2_4)#(None, 256, 256, 32)
        conv1_5=torch.cat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4],1)#(None, 256, 256, 160
        conv1_5=self.standard_unit_160_0(conv1_5)

        nestnet_output_1=self.conv(conv1_2)
        nestnet_output_2 = self.conv2(conv1_3)
        nestnet_output_3 = self.conv3(conv1_4)
        nestnet_output_4 = self.conv4(conv1_5)
        conv_fuse=torch.cat([conv1_2, conv1_3, conv1_4, conv1_5],1)
        nestnet_output_5=self.conv5(conv_fuse)
        # return nestnet_output_1,nestnet_output_2,nestnet_output_3,nestnet_output_4,nestnet_output_5

        return nestnet_output_5,nestnet_output_4
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

class SiamUnet_diff(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self):
        super(SiamUnet_diff, self).__init__()

        # self.input_nbr = input_nbr
        # print('SiamUnet_diff')
        self.conv11 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, 2, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1[:,:3,:,:]))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p_1 = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2[:,:3,:,:]))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p_2 = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        # Stage 4d
        x4d = self.upconv4(torch.abs(x4p_1 - x4p_2))
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        # return self.sm(x11d)
        return x11d,x11d
cfg.TRAINLOG.NETWORK_DICT = {"Unet": Unet,
                             "Nest_Net":Nest_Net,
                             "DSIFN":DSIFN,
                            "UCDNet":UCDNet,
                             "SiamUnet_diff":SiamUnet_diff,
                            "UCDNet_ASSPMultiout":UCDNet_ASSPMultiout,
                            "UCDNetRes":UCDNetRes,
                            "DeepLab":DeepLab,
                            "FCSiamDiffBNnewMultiPSPBN":FCSiamDiffBNnewMultiPSPBN,
                            "FCSiamDiff":FCSiamDiff,
                             "FCSiamConc":FCSiamConc,
                            "Model_FullMM":Model_FullMM,
                            "Model_PartialMM":Model_PartialMM,
                            "HLCDNetlog":HLCDNetlog,
                            "HLCDNet2":HLCDNet2,
                            "HLCDNetS":HLCDNetS,#best 3-20
                            "HLCDNetNo":HLCDNetNo,
                             "HLCDNetNoM0":HLCDNetNoM0,
                             "HLCDNetG":HLCDNetG,
                             "HLCDNetGB":HLCDNetGB,
                            "HLCDNetS2":HLCDNetS2,
                             "HLCDNetSL":HLCDNetSL,
                             "HLCDNetSLS":HLCDNetSLS,
                             "HLCDNetSS":HLCDNetSS,#best 3-22
                             "HLCDNetSSC":HLCDNetSSC,
                             #######DA+BN
                             "HLCDNetSBN":HLCDNetSBN

                             }