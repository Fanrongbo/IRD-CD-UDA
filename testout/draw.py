# import numpy as np
# h=3
# w=4
# xx=np.arange(h)
# yy=np.arange(w)
# xx=np.expand_dims(xx,axis=0)
# xx=np.expand_dims(xx.repeat(w,axis=0),axis=-1)
# yy=np.expand_dims(yy,axis=0)
# yy=np.expand_dims(yy.repeat(h,axis=0).transpose(1,0),axis=-1)
#
# print(yy.shape,xx.shape)
# pixel=np.concatenate([xx,yy],axis=-1)
# pixel=np.fla
# print(pixel)
#
# xx=np.expand_dims(xx.reshape((h,w)),axis=0)
#
# print(xx.shape)
#
# A_expand = xx.unsqueeze(1).expand(xx, xx, feat_len)
# B_expand = B.unsqueeze(0).expand(xx, bs_T, feat_len)
import torch
h=3
w=3
c=5
cc= torch.arange(c)
xx = torch.arange(h)
yy = torch.arange(w)
x_expand = xx.unsqueeze(1).expand(5, h, w).unsqueeze(-1)
y_expand = yy.unsqueeze(0).expand(5, h, w).unsqueeze(-1)
# p = torch.cat([x_expand, y_expand], dim=-1)
# print(p.shape)
aa=torch.ones((h,w)).unsqueeze(0).unsqueeze(-1)
cc=torch.cat([aa, aa*2,aa*3,aa*4,aa*5], dim=0)
# print(cc.shape)
p = torch.cat([x_expand, y_expand,cc], dim=-1)
#
print(p)
# print(p
#       )

p = torch.flatten(p, start_dim=0, end_dim=2)
