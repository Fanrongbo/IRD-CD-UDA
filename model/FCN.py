from model.resnet2 import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,SynchronizedBatchNorm2d,resnet101_diff
import torch.nn.functional as F

import torch
import torch.nn as nn
def build_backbone(backbone, out_stride=32, mult_grid=False):
    if backbone == 'resnet18':
        return resnet18(out_stride, mult_grid)
    elif backbone == 'resnet34':
        return resnet34(out_stride, mult_grid)
    elif backbone == 'resnet50':
        return resnet50(out_stride, mult_grid)
    elif backbone == 'resnet101':
        return resnet101(out_stride, mult_grid)

    else:
        raise NotImplementedError
class FCN_ResNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet18', out_stride=32, mult_grid=False):
        backbone = 'resnet50'

        super(FCN_ResNet, self).__init__()

        if backbone == 'resnet18' or backbone == 'resnet34':
            expansion = 1
        elif backbone == 'resnet50' or backbone == 'resnet101':
            expansion = 4
        self.backbone = build_backbone(backbone, out_stride, mult_grid)
        # self.backbone=ResNet(in_channels=3, output_stride=out_stride, backbone=backbone,pretrained=False, AG_flag=False)
        # print('&***&&&&&&&&&&&&&&&&&&&&&&&&')
        self.conv_1 = nn.Conv2d(in_channels=512 * expansion, out_channels=512 * expansion // 4, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=256 * expansion, out_channels=num_classes, kernel_size=1)
        self.conv_3 = nn.Conv2d(in_channels=128 * expansion, out_channels=num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(in_channels=64 * expansion, out_channels=num_classes, kernel_size=1)

        self._init_weight()

    def forward(self, x1, x2):
        layers_1 = self.backbone(x1[:,:3,:,:])  # resnet 4 layers
        layers_2 = self.backbone(x2[:,:3,:,:])  # resnet 4 layers
        diff3 = torch.abs(layers_1[3] - layers_2[3])
        diff2 = torch.abs(layers_1[2] - layers_2[2])
        diff1 = torch.abs(layers_1[1] - layers_2[1])
        diff0 = torch.abs(layers_1[0] - layers_2[0])

        layers3 = self.conv_1(diff3)
        layers3 = F.interpolate(layers3, diff2.size()[2:], mode="bilinear", align_corners=True)
        layers2 = self.conv_2(diff2)

        output = layers2 + layers3
        output = F.interpolate(output, diff1.size()[2:], mode="bilinear", align_corners=True)
        layers1 = self.conv_3(diff1)

        output = output + layers1
        output = F.interpolate(output, diff0.size()[2:], mode="bilinear", align_corners=True)
        layers0 = self.conv_4(diff0)

        output = output + layers0
        output = F.interpolate(output, x1.size()[2:], mode="bilinear", align_corners=True)

        return output, output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.conv_1, self.conv_2, self.conv_3, self.conv_4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p