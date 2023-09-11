import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.padding import ReplicationPad2d
# from   models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,SynchronizedBatchNorm2d,resnet101_diff
import math
class ConvBlock(nn.Module):
    def __init__(self,dim_in,dim_feature):
        super(ConvBlock, self).__init__()
        self.conv=nn.Conv2d(dim_in, dim_feature, kernel_size=7, padding=3,stride=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_feature)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.max_pool(x)
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
        residual = x
        residual = self.convplus(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DSAMNet(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(DSAMNet, self).__init__()
        hideen_num=[64,128,256,512]
        total_hiddennum=0
        for num in hideen_num:
            total_hiddennum=total_hiddennum+num
        self.total_hiddennum=total_hiddennum
        self.ConvBlock_layer=ConvBlock(in_dim,32)
        self.BasicBlock1 = BasicBlock(32, hideen_num[0],1,2)#128,128,64
        self.BasicBlock2 = BasicBlock(hideen_num[0], hideen_num[1])#128,128,128
        self.BasicBlock3 = BasicBlock(hideen_num[1], hideen_num[2])#128,128,256
        self.BasicBlock4 = BasicBlock(hideen_num[2], hideen_num[3],1,2)#64,64,512
        self.conv1=nn.Conv2d(hideen_num[0], 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(hideen_num[1], 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(hideen_num[2], 32, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(hideen_num[3], 32, kernel_size=1, bias=False)
        featuresize=32
        self.conv5 = nn.Conv2d(32*4, featuresize, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(featuresize)
        self.relu = nn.ReLU()
        self.conv6 = nn.Conv2d(featuresize, featuresize, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(featuresize)

        self.up2=nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.DS1conv=nn.Conv2d(hideen_num[0], hideen_num[0], kernel_size=3, bias=False)
        self.DS11up = nn.ConvTranspose2d(hideen_num[0], hideen_num[0]//2, kernel_size=2, stride=2)
        self.DS12up = nn.ConvTranspose2d(hideen_num[0]//2, 1, kernel_size=2, stride=2)
        # self.DS2conv = nn.Conv2d(hideen_num[1], hideen_num[1], kernel_size=3, bias=False)
        self.DS21up = nn.ConvTranspose2d(hideen_num[1], hideen_num[1]//2, kernel_size=2, stride=2)
        self.DS22up = nn.ConvTranspose2d(hideen_num[1]//2, 1, kernel_size=2, stride=2)

        self.metric1up = nn.ConvTranspose2d(featuresize, featuresize, kernel_size=2, stride=2)
        self.sigmoid=nn.Sigmoid()
    def feature_extract(self,x):
        x=self.ConvBlock_layer(x)#128,128,32
        # print('x',x.shape)
        feature_1=self.BasicBlock1(x)#128,128,64
        feature_2=self.BasicBlock2(feature_1)#128,128,128
        feature_3=self.BasicBlock3(feature_2)#128,128,256
        feature_4=self.BasicBlock4(feature_3)#64,64,512
        return feature_1,feature_2,feature_3,feature_4
    def metric_Moudule(self,feature_1, feature_2, feature_3, feature_4):
        feature_1 = self.conv1(feature_1)#64,64,32
        feature_1 = self.up2(feature_1)  # 128,128,32
        feature_2 = self.conv2(feature_2)#32,32,32
        feature_2 = self.up2(feature_2)#128,128,32
        feature_3 = self.conv3(feature_3)#32,32,32
        feature_3 = self.up2(feature_3)#128,128,32
        feature_4 = self.conv4(feature_4)#64,64,32
        feature_4 = self.up4(feature_4)#128,128,32
        feature=torch.cat([feature_1, feature_2, feature_3, feature_4],1)

        # feature = self.up(feature)#128,128,32
        feature = self.conv5(feature)
        # print('feature', feature.shape)

        feature = self.bn5(feature)
        feature = self.relu(feature)

        feature = self.conv6(feature)
        feature=self.metric1up(feature)#256,256,32
        # print('feature2', feature.shape)

        # feature = self.bn6(feature)
        # feature = self.relu(feature)
        return feature
    def DS(self,feature_1_diff,feature_2_diff):
        feature_1_diff=self.DS11up(feature_1_diff)
        feature_1_diff = self.DS12up(feature_1_diff)
        # feature_1_diff=self.sigmoid(feature_1_diff)
        feature_2_diff = self.DS21up(feature_2_diff)
        feature_2_diff = self.DS22up(feature_2_diff)
        # feature_2_diff = self.sigmoid(feature_2_diff)
        return feature_1_diff,feature_2_diff
    def forward(self, x1,x2):

        feature_11, feature_12, feature_13, feature_14 = self.feature_extract(x1)
        feature_21, feature_22, feature_23, feature_24 = self.feature_extract(x2)

        feature_T1 = self.metric_Moudule(feature_11, feature_12, feature_13, feature_14)
        feature_T2 = self.metric_Moudule(feature_21, feature_22, feature_23, feature_24)
        # print('feature_T1', feature_T1.shape)
        feature_1_diff=torch.abs(feature_11-feature_21)
        feature_2_diff = torch.abs(feature_12 - feature_22)
        feature_1_diff, feature_2_diff=self.DS(feature_1_diff,feature_2_diff)#DiceLoss to label

        diff=feature_T1.mean(dim=1)-feature_T2.mean(dim=1)
        # dist=torch.sqrt(diff*diff)#contrastive loss for optimization  BCLLOSS to label
        dist=self.sigmoid(diff)
        # print('diff', diff.shape)
        return [dist,feature_1_diff, feature_2_diff],dist

class DSAMNet2(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(DSAMNet2, self).__init__()
        hideen_num=[64,128,256,512]
        total_hiddennum=0
        for num in hideen_num:
            total_hiddennum=total_hiddennum+num
        self.total_hiddennum=total_hiddennum
        self.ConvBlock_layer=ConvBlock(in_dim,32)
        self.BasicBlock1 = BasicBlock(32, hideen_num[0],1,2)#128,128,64
        self.BasicBlock2 = BasicBlock(hideen_num[0], hideen_num[1])#128,128,128
        self.BasicBlock3 = BasicBlock(hideen_num[1], hideen_num[2])#128,128,256
        self.BasicBlock4 = BasicBlock(hideen_num[2], hideen_num[3],1,2)#64,64,512
        self.conv1=nn.Conv2d(hideen_num[0], 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(hideen_num[1], 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(hideen_num[2], 32, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(hideen_num[3], 32, kernel_size=1, bias=False)
        featuresize=32
        self.conv5 = nn.Conv2d(32*4, featuresize, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(featuresize)
        self.relu = nn.ReLU()
        self.conv6 = nn.Conv2d(featuresize, featuresize, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(featuresize)

        self.up2=nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.DS1conv=nn.Conv2d(hideen_num[0], hideen_num[0], kernel_size=3, bias=False)
        self.DS11up = nn.ConvTranspose2d(hideen_num[0], hideen_num[0]//2, kernel_size=2, stride=2)
        self.DS12up = nn.ConvTranspose2d(hideen_num[0]//2, 2, kernel_size=2, stride=2)
        # self.DS2conv = nn.Conv2d(hideen_num[1], hideen_num[1], kernel_size=3, bias=False)
        self.DS21up = nn.ConvTranspose2d(hideen_num[1], hideen_num[1]//2, kernel_size=2, stride=2)
        self.DS22up = nn.ConvTranspose2d(hideen_num[1]//2, 2, kernel_size=2, stride=2)

        self.metric1up = nn.ConvTranspose2d(featuresize, featuresize, kernel_size=2, stride=2)
        self.convout = nn.Conv2d(featuresize, 1, kernel_size=1, bias=False)
        self.sigmoid=nn.Sigmoid()
    def feature_extract(self,x):
        x=self.ConvBlock_layer(x)#128,128,32
        # print('x',x.shape)
        feature_1=self.BasicBlock1(x)#128,128,64
        feature_2=self.BasicBlock2(feature_1)#128,128,128
        feature_3=self.BasicBlock3(feature_2)#128,128,256
        feature_4=self.BasicBlock4(feature_3)#64,64,512
        return feature_1,feature_2,feature_3,feature_4
    def metric_Moudule(self,feature_1, feature_2, feature_3, feature_4):
        feature_1 = self.conv1(feature_1)#64,64,32
        feature_1 = self.up2(feature_1)  # 128,128,32
        feature_2 = self.conv2(feature_2)#32,32,32
        feature_2 = self.up2(feature_2)#128,128,32
        feature_3 = self.conv3(feature_3)#32,32,32
        feature_3 = self.up2(feature_3)#128,128,32
        feature_4 = self.conv4(feature_4)#64,64,32
        feature_4 = self.up4(feature_4)#128,128,32
        feature=torch.cat([feature_1, feature_2, feature_3, feature_4],1)

        # feature = self.up(feature)#128,128,32
        feature = self.conv5(feature)
        # print('feature', feature.shape)

        feature = self.bn5(feature)
        feature = self.relu(feature)

        feature = self.conv6(feature)
        feature=self.metric1up(feature)#256,256,32
        # print('feature2', feature.shape)

        # feature = self.bn6(feature)
        # feature = self.relu(feature)
        return feature
    def DS(self,feature_1_diff,feature_2_diff):
        feature_1_diff=self.DS11up(feature_1_diff)
        feature_1_diff = self.DS12up(feature_1_diff)
        # feature_1_diff=self.sigmoid(feature_1_diff)
        feature_2_diff = self.DS21up(feature_2_diff)
        feature_2_diff = self.DS22up(feature_2_diff)
        # feature_2_diff = self.sigmoid(feature_2_diff)
        return feature_1_diff,feature_2_diff
    def forward(self, x1,x2):

        feature_11, feature_12, feature_13, feature_14 = self.feature_extract(x1[:,:3,:,:])
        feature_21, feature_22, feature_23, feature_24 = self.feature_extract(x2[:,:3,:,:])

        feature_T1 = self.metric_Moudule(feature_11, feature_12, feature_13, feature_14)
        feature_T2 = self.metric_Moudule(feature_21, feature_22, feature_23, feature_24)
        # print('feature_T1', feature_T1.shape)
        feature_1_diff=torch.abs(feature_11-feature_21)
        feature_2_diff = torch.abs(feature_12 - feature_22)
        feature_1_diff, feature_2_diff=self.DS(feature_1_diff,feature_2_diff)#DiceLoss to label256,256,32

        # diff=feature_T1-feature_T2
        # diff=self.convout(diff)
        diff = feature_T1.mean(dim=1) - feature_T2.mean(dim=1)
        # dist=torch.sqrt(diff*diff)#contrastive loss for optimization  BCLLOSS to label
        # dist = self.sigmoid(diff)
        dist=torch.sqrt(diff*diff)#contrastive loss for optimization  BCLLOSS to label
        return [dist,feature_1_diff, feature_2_diff],diff

def CD_MarginRankingLoss(dist, target, margin=.5):
    # loss1=0
    # loss2=0
    # # for x in zip(dist):
    # for i in range(dist.shape[1]):
    #     loss1=loss1+(1-target)*torch.pow(dist[:,i,:,:],2)
    #     zeros = torch.zeros_like(dist[:,i,:,:])
    #     margin_out=torch.pow(dist[:,i,:,:]-margin,2)
    #     margin_out = torch.where(margin_out > 0, margin_out, zeros)
    #     loss2=loss2+target*margin_out
    # print('dist', dist.dtype)

    loss1=(1-target)*torch.pow(dist,2)
    zeros = torch.zeros_like(dist)
    margin_out=torch.pow(dist-margin,2)
    margin_out = torch.where(margin_out > 0, margin_out, zeros)
    loss2=target*margin_out
    loss=loss1+loss2
    loss = loss.mean()

    return loss


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)
import os
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = DSAMNet(3,1).cuda()
    # modela=model(6,2)
    # summary(model, (6, 256, 256), device="cpu")
    # print(model)
    target = torch.randn([8, 256, 256]).cuda()

    ones = torch.ones_like(target)
    zeros = torch.zeros_like(target)
    target = torch.where(target > 0.5, ones, zeros).cuda()
    print('target',target.dtype)

    # model = nn.DataParallel(model).cuda()
    total = sum([param.nelement() for param in model.parameters()])

    for mouble_name, parameters in model.named_parameters():
        print(mouble_name, ':', parameters.size())
    print('Param Num=', total)
    input1 = torch.randn([8, 3, 256, 256], requires_grad=True).cuda()
    input2 = torch.randn([8, 3, 256, 256], requires_grad=True).cuda()
    cd_preds,cd_pred=model(input1,input2)
    print('out:',cd_preds[0].shape,cd_preds[1].shape,cd_preds[2].shape)
    criterion=[CD_MarginRankingLoss,dice_loss]
    margin_loss = criterion[0](cd_preds[0], target)
    dice_loss_1 = criterion[1](cd_preds[1], target.long().unsqueeze(1))
    dice_loss_2 = criterion[1](cd_preds[2], target.long().unsqueeze(1))
    print(margin_loss,dice_loss_1,dice_loss_2)



    target = torch.randn([1, 3, 3]).cuda()
    ones = torch.ones_like(target)
    zeros = torch.zeros_like(target)
    target = torch.where(target > 0.5, ones, zeros).cuda().long().unsqueeze(1)
    print('target', target,target.dtype)
    true_1_hot = torch.eye(2)[target.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    print('true_1_hot', true_1_hot[0,0,:,:],true_1_hot[0,1,:,:], true_1_hot.shape)


    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # model = Nest_Net(6, 2)
    # # modela=model(6,2)
    # summary(model, (6, 256, 256), device="cpu")
    # print(model)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
