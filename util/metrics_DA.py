import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.Weight import Weight
class NegativeLearningLoss(nn.Module):
    def __init__(self, threshold=0.05):
        super(NegativeLearningLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        mask = (predict < self.threshold).detach()
        negative_loss_item = -1 * mask * torch.log(1 - predict + 1e-6)
        negative_loss = torch.sum(negative_loss_item) / torch.sum(mask)

        return negative_loss

class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def CORAL(self,source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        # print('source',source.shape)#18, 1024]
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss

    def selecdata(self,feature, label):
        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((1, 0, 2, 3)), start_dim=1, end_dim=3)
        label_index = torch.nonzero(label_flatten)

        label_index = torch.flatten(label_index)
        label_index_rand = torch.randperm(label_index.nelement())
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[:, label_index[0]].unsqueeze(0)
        return feature_flatten_select, label_index, feature_flatten
    def forward(self, source, target,label_source,pred_target):
        chgthreshold=800
        unchgthreshold=800
        H, W = source.size(2), source.size(3)
        label_source = F.interpolate(label_source.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        pred_target = F.interpolate(pred_target.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False)
        ones = torch.ones_like(label_source)
        zeros = torch.zeros_like(label_source)
        label_source = torch.where(label_source > 0.5, ones, zeros)
        pred_target = torch.where(pred_target > 0.5, ones, zeros)
        ############### change origin
        # source = (label_source.repeat([1, source.shape[1], 1, 1])).float()
        # target = (pred_target.repeat([1, target.shape[1], 1, 1])).float()
        source_chg_flatten_select,source_chg_index,source_chg_flatten=self.selecdata(source,label_source)
        target_chg_flatten_select,target_chg_index,target_chg_flatten=self.selecdata(target,pred_target)
        # one=torch.ones_like(source_chg_flatten[:,1])

        # print('source_chg_flatten_select',source_chg_flatten_select.shape)
        if source_chg_index.shape[0]<chgthreshold or target_chg_index.shape[0]<chgthreshold:
            chgthreshold= np.minimum(source_chg_index.shape[0],target_chg_index.shape[0])
            # print('chgthreshold',chgthreshold)
        source_chg_flatten_select=source_chg_flatten[:,source_chg_index[0:chgthreshold]]
        target_chg_flatten_select=target_chg_flatten[:, target_chg_index[0:chgthreshold]]


        ###############################################
        ######################unchange
        # source = ((1-label_source).repeat([1, source.shape[1], 1, 1])).float()
        # target = ((1-pred_target).repeat([1, target.shape[1], 1, 1])).float()
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1-label_source)
        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1-pred_target)
        # one = torch.ones_like(source_unchg_flatten[:, 1])

        # print('source_unchg_flatten_select', source_unchg_flatten_select.shape)
        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        source_unchg_flatten_select=source_unchg_flatten[:, source_unchg_index[0:unchgthreshold]]
        target_unchg_flatten_select=target_unchg_flatten[:, target_unchg_index[0:unchgthreshold]]
        CORAL_value_chg = self.CORAL(source_chg_flatten_select, target_chg_flatten_select)
        CORAL_value_unchg = self.CORAL(source_unchg_flatten_select, target_unchg_flatten_select)

        return CORAL_value_chg + CORAL_value_unchg

def CORAL_ori(source, target):
    source = torch.flatten(source, start_dim=1, end_dim=3)[0:2,:]
    target = torch.flatten(target, start_dim=1, end_dim=3)[0:2,:]

    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=2):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        # print(f_of_X.shape,f_of_Y.shape)
        # f_of_X=torch.flatten(f_of_X,start_dim=1,end_dim=3)
        # f_of_Y=torch.flatten(f_of_Y,start_dim=1,end_dim=3)

        delta = (f_of_X.float().mean(0) - f_of_Y.float().mean(0))
        # print(delta.shape)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):

        if self.kernel_type == 'linear':
            linear_mmd2_value=self.linear_mmd2(source, target)
            return  linear_mmd2_value
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss






class SelecFeat():
    def selecdata(self, feature, label):
        #label 6, 1, 128, 128
        #feature 6, 32, 128, 128

        label_flatten = torch.flatten(label.squeeze(1), start_dim=0, end_dim=2)
        feature_flatten = torch.flatten(feature.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)
        label_index = torch.nonzero(label_flatten)
        label_index = torch.flatten(label_index)
        # print('label_index',label_index.shape,label.sum())
        label_index_rand = torch.randperm(label_index.nelement())
        # print('label_index.nelement()',label_index.nelement(),label_index_rand)
        label_index = label_index[label_index_rand]
        feature_flatten_select = feature_flatten[label_index,:]#bs,c
        # print('feature_flatten_select.nelement()', feature_flatten_select.shape,label_flatten.sum())
        return feature_flatten_select, label_index, feature_flatten
    def to_onehot(self,label, num_classes):
        identity = (torch.eye(num_classes)).to(self.device)
        onehot = torch.index_select(identity, 0, label)
        return onehot
    def select_featureS(self,source,s_label):
        chgthreshold = 2400  # select 1000 pixel
        unchgthreshold = 2400
        self.chgthreshold = chgthreshold
        self.unchgthreshold = unchgthreshold
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, s_label)
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - s_label)
        # source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold], :]  # bs,c
        # source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        source_unchg_flatten_select = source_unchg_flatten_select[0:unchgthreshold]  # bs,c
        source_chg_flatten_select = source_chg_flatten_select[0:chgthreshold]  # bs,c
        return source_chg_flatten_select, source_unchg_flatten_select
    def position(self,device):
        h=256
        w=256
        xx = torch.arange(h)
        yy = torch.arange(w)
        x_expand = xx.unsqueeze(1).expand(5,h, w).unsqueeze(-1)
        y_expand = yy.unsqueeze(0).expand(5,h, w).unsqueeze(-1)
        aa = torch.ones((h, w)).unsqueeze(0).unsqueeze(-1)
        cc = torch.cat([aa, aa * 2, aa * 3, aa * 4, aa * 5], dim=0)
        p = torch.cat([x_expand, y_expand,cc], dim=-1)

        p = torch.flatten(p, start_dim=0, end_dim=2)
        # print(p)

        # p = p.unsqueeze(0).expand(5,0)
        # print('p',p.shape)
        return p
        # print(yy.shape, xx.shape)

    def select_featureST(self,source,s_label,target,pseudo_label,softmaxLabel,softLog,p=0,pe=0,device='cuda'):

        # pp=self.position(device)

        # print(pp.shape,source.shape)

        # pp=pp.to(device)
        self.device=device
        chgthreshold = 1200 # select 1000 pixel
        unchgthreshold = 1200
        self.chgthreshold=chgthreshold
        self.unchgthreshold=unchgthreshold
        pseudo_label=pseudo_label.unsqueeze(1)
        # print('softmaxLabel',softmaxLabel.shape)#[13, 2, 65536]
        # softmaxLabelori=softmaxLabel.reshape(-1,2,s_label.shape[2],s_label.shape[3])#[bs,2,h,w]->[bs,h,w,2]
        softmaxLabelori=softmaxLabel
        # print('softmaxLabelori',softmaxLabelori.shape,pseudo_label.shape)
        self.uu = (softmaxLabel[:, 0, :, :] * (1 - pseudo_label.squeeze(1))).sum() / ((1 - pseudo_label).sum() + 1)
        self.cc = ((softmaxLabel[:, 1, :, :]) * pseudo_label.squeeze(1)).sum() / (pseudo_label.sum() + 1)

        softmaxLabel = torch.flatten(softmaxLabelori.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2

        # softLogS = softLogS.reshape(-1, 2, s_label.shape[2], s_label.shape[3])  # [bs,2,h,w]->[bs,h,w,2]
        # softLogS = torch.flatten(softLogS.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2
        # softLogT = softLogT.reshape(-1, 2, s_label.shape[2], s_label.shape[3])  # [bs,2,h,w]->[bs,h,w,2]
        # softLogT = torch.flatten(softLogT.permute((0, 2, 3, 1)), start_dim=0, end_dim=2)  # bs,2
######################change
        source_chg_flatten_select, source_chg_index, source_chg_flatten = self.selecdata(source, s_label)
        ones=torch.ones_like(pseudo_label)
        zeros=torch.zeros_like(pseudo_label)

        pseudo_labeltChg=torch.where(softmaxLabelori[:,1,:,:].unsqueeze(1)>(p-pe),pseudo_label,zeros).detach()############################3############################3############################3############################3############################3
        # pseudo_labeltChg=torch.where((softmaxLabelori[:,1,:,:]/softmaxLabelori[:,1,:,:].max()).unsqueeze(1)>p,pseudo_label,zeros)
        # print('pseudo_label',pseudo_labeltChg.shape,pseudo_label.shape,pseudo_label.sum(),pseudo_labeltChg.sum())
        target_chg_flatten_select, target_chg_index, target_chg_flatten = self.selecdata(target, pseudo_labeltChg)


        if source_chg_index.shape[0] < chgthreshold or target_chg_index.shape[0] < chgthreshold:
            chgthreshold = np.minimum(source_chg_index.shape[0], target_chg_index.shape[0])
        source_chg_flatten_select = source_chg_flatten[source_chg_index[0:chgthreshold],:]#bs,c
        target_chg_flatten_select = target_chg_flatten[target_chg_index[0:chgthreshold],:]#bs,c
        # print(pp.shape,target_chg_flatten.shape)
        # target_pchg=pp[target_chg_index[0:chgthreshold].cpu(),:].unsqueeze(0)
        # print('target_p',target_p)
        softmaxLabel_chg_select = softmaxLabel[target_chg_index[0:chgthreshold]]  # [bs,2]

        # softLogT_chg_select = softLogT[target_chg_index[0:chgthreshold]]  # [bs,2]
        # softLogS_chg_select = softLogS[source_chg_index[0:chgthreshold]]
        # print(softmaxLabel_chg_select)
        # print('softmaxLabel_chg_select',softmaxLabel_chg_select.shape)
        # target_chg_flatten_selectW = target_chg_flatten_select * softmaxLabel_chg_select[:, 1].unsqueeze(1)
####################unchg
        source_unchg_flatten_select, source_unchg_index, source_unchg_flatten = self.selecdata(source, 1 - s_label)
        # print('softmaxLabel',softmaxLabel.shape)
        pseudo_labeltunChg = torch.where(softmaxLabelori[:,0,:,:].unsqueeze(1)>p, pseudo_label, ones).detach()############################3############################3############################3############################3
        # pseudo_labeltunChg = torch.where((softmaxLabelori[:,0,:,:]/softmaxLabelori[:,0,:,:].max()).unsqueeze(1)>p, pseudo_label, ones)


        target_unchg_flatten_select, target_unchg_index, target_unchg_flatten = self.selecdata(target, 1 - pseudo_labeltunChg)


        if source_unchg_index.shape[0] < unchgthreshold or target_unchg_index.shape[0] < unchgthreshold:
            unchgthreshold = np.minimum(source_unchg_index.shape[0], target_unchg_index.shape[0])
        if unchgthreshold > chgthreshold:
            unchgthreshold = chgthreshold
        source_unchg_flatten_select = source_unchg_flatten[source_unchg_index[0:unchgthreshold], :]  # bs,c
        # softLogS_unchg_select=softLogS[source_unchg_index[0:unchgthreshold]]

        target_unchg_flatten_select = target_unchg_flatten[target_unchg_index[0:unchgthreshold], :]  # bs,c
        # target_punchg = pp[target_unchg_index[0:unchgthreshold].cpu(), :].unsqueeze(0)
        softmaxLabel_unchg_select = softmaxLabel[target_unchg_index[0:unchgthreshold]]
        # softLogT_unchg_select = softLogT[target_unchg_index[0:unchgthreshold]]
        # print('target_pchg',target_pchg.shape,target_punchg.shape)
        # target_unchg_flatten_selectW = target_unchg_flatten_select * softmaxLabel_unchg_select[:, 0].unsqueeze(1)#weight
        self.chgNum = chgthreshold
        self.unchgNum = unchgthreshold
        unchglabel = self.to_onehot(torch.zeros_like(softmaxLabel_unchg_select[:, 0]).long(), 2)
        chglabel = self.to_onehot(torch.ones_like(softmaxLabel_unchg_select[:, 1]).long(), 2)
        # print(unchglabel, chglabel)
        # print('s',softmaxLabel_unchg_select.shape,softmaxLabel_chg_select[1].shape,softmaxLabel_unchg_select[:,0].min(),softmaxLabel_chg_select[:,1].min())
        s_label_select = torch.cat([unchglabel,chglabel], dim=0).detach()
        # print('s_label_select',s_label_select)
        t_label_select = torch.cat([softmaxLabel_unchg_select,softmaxLabel_chg_select ], dim=0).detach()
        # print('softmaxLabel_unchg_select',t_label_select.shape)
        t_label_select2=torch.cat([softmaxLabel_unchg_select,softmaxLabel_chg_select ], dim=0).detach()
        # softLogS=torch.cat([softLogS_unchg_select,softLogS_chg_select], dim=0)
        # softLogT=torch.cat([softLogT_unchg_select,softLogT_chg_select], dim=0)

        # print('t_label_select2',t_label_select2.shape)
        return source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select, \
               s_label_select, t_label_select, t_label_select2, []
        # return source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select,\
        #        s_label_select,t_label_select,t_label_select2,torch.cat([target_pchg,target_punchg],dim=0)