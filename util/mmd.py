import torch
from torch import nn
from utils.utils import to_cuda

class MMD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul,
		gammas=[], joint=False,device=None, **kwargs):
        self.device=device
        self.num_layers = num_layers
        # assert(len(kernel_num) == num_layers and len(kernel_mul) == num_layers), \
        #     "Each layer's kernel choice should be specified."

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.gammas = gammas

        self.joint = joint

    def compute_paired_dist(self, A, B):

        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        # print(A_expand.shape)#[2400, 2400, 32]

        dist=F.cosine_similarity(A_expand,B_expand,dim=2)
        # dist = F.pairwise_distance(A_expand, B_expand, p=2)
        # print('dist',dist.shape,dist.max(),dist.min(),A_expand.shape,B_expand.shape)
        # dist = (((A_expand - B_expand)) ** 2).sum(2)
        # dist=dist/(dist.max()+0.001)
        # print('dist2', dist.shape, dist.max(), dist.min())

        return dist

    def compute_weight(self, A, B):

        # bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = B.size(1)

        # A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_T, bs_T, feat_len)
        # print(A_expand.shape)#[2400, 2400, 32]
        # print(A_expand.shape)
        weight = B_expand
        # dist = F.pairwise_distance(A_expand, B_expand, p=2)
        # print('dist',dist.shape,dist.max(),dist.min(),A_expand.shape,B_expand.shape)
        # dist = (((A_expand - B_expand)) ** 2).sum(2)
        # dist=dist/(dist.max()+0.001)
        # print('dist2', dist.shape, dist.max(), dist.min())

        return weight
    def gamma_estimation2(self, dist):
        dist_sum = torch.sum(dist['st'])

        bs_S = dist['st'].size(0)
        bs_T = dist['st'].size(1)
        # print(bs_S,bs_T)
        N =  bs_S * bs_T - bs_S - bs_T+1
        gamma = dist_sum.item() / N
        return gamma

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
	    	2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N
        return gamma

    def gamma_estimation2(self, dist):
        dist_sum = 2 * torch.sum(dist['st'])

        bs_S = dist['st'].size(0)
        bs_T = dist['st'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N
        return gamma
    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        # gamma_tensor = to_cuda(torch.tensor(gamma_list))
        gamma_tensor = (torch.tensor(gamma_list)).to(self.device)
        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        dist = dist.unsqueeze(0) / gamma_tensor.view(-1, 1, 1)
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers):
        kernel_dist = {}
        num_layers = self.num_layers
        for i in range(num_layers):

            dist = dist_layers[i]
            gamma = gamma_layers[i]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if len(kernel_dist.keys()) == 0:
                kernel_dist = {key: self.compute_kernel_dist(dist[key],
			gamma, cur_kernel_num, cur_kernel_mul)
                        for key in ['ss', 'tt', 'st']}
                continue

            if not self.joint:
                kernel_dist = {key: kernel_dist[key] +
		    	self.compute_kernel_dist(dist[key], gamma, cur_kernel_num, cur_kernel_mul)
                        for key in ['ss', 'tt', 'st']}

            else:
                kernel_dist = {key: kernel_dist[key] *
		    	self.compute_kernel_dist(dist[key], gamma, cur_kernel_num, cur_kernel_mul)
                        for key in ['ss', 'tt', 'st']}

        return kernel_dist

    def kernel_layer_aggregation2(self, dist_layers, gamma_layers):
        kernel_dist = {}
        num_layers = self.num_layers
        for i in range(num_layers):

            dist = dist_layers[i]
            gamma = gamma_layers[i]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if len(kernel_dist.keys()) == 0:
                kernel_dist = {key: self.compute_kernel_dist(dist[key],
                                                             gamma, cur_kernel_num, cur_kernel_mul)
                               for key in ['st']}
                continue

            if not self.joint:
                kernel_dist = {key: kernel_dist[key] +
                                    self.compute_kernel_dist(dist[key], gamma, cur_kernel_num, cur_kernel_mul)
                               for key in ['st']}

            else:
                kernel_dist = {key: kernel_dist[key] *
                                    self.compute_kernel_dist(dist[key], gamma, cur_kernel_num, cur_kernel_mul)
                               for key in ['st']}

        return kernel_dist

    def forward(self, source, target,s_label_select,t_label_select, **kwargs):

        assert(len(source) == self.num_layers and len(target) == self.num_layers), \
            "The length of source (%d) or target (%d) doesn't match with %d, please check." \
            % (len(source), len(target), self.num_layers)

        assert len(source) != 0, "No data provided."

        dist_layers = []
        gamma_layers = []
        for i in range(self.num_layers):

            cur_source = source[i]
            cur_target = target[i]#e([2400, 32])
            # print(cur_source.shape)
            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)#[2n,2n]
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)
            # print('\n')

            dist_layers += [dist]
            if len(self.gammas) > i and self.gammas[i] is not None:
                gamma_layers += [self.gammas[i]]
            else:
                gamma_layers += [self.gamma_estimation(dist)]
        # dagt={}
        num=dist['st'].shape[0]//2
        # kernel_dist = self.kernel_layer_aggregation(dist_layers, gamma_layers)

        kernel_dist = self.kernel_layer_aggregation2(dist_layers, gamma_layers)#st
        if True:
            weight={}
            t_label_select_u=t_label_select[:num,0].unsqueeze(1)
            t_label_select_c = t_label_select[num:, 1].unsqueeze(1)
            self.maxu=t_label_select_u.mean()
            self.maxc = t_label_select_c.mean()
            # print('t_label_select_u',t_label_select_u.mean(),t_label_select_c.mean())
            t_label_select=torch.cat([t_label_select_u,t_label_select_c],dim=0)#
            s_label_select=torch.ones_like(t_label_select)
            # print('t_label_select_u', t_label_select.shape)
            # print('t_label_select_u', t_label_select_u.shape, t_label_select_u)
            # weight['tt']=torch.matmul(t_label_select,t_label_select.permute(1,0))
            # weight['ss'] = torch.matmul(s_label_select, s_label_select.permute(1, 0))
            # weight['st']=self.compute_weight(0, t_label_select)

            weight['st'] = torch.matmul(s_label_select, t_label_select.permute(1, 0)).detach()
            # weight['st'] = torch.matmul(t_label_select, s_label_select.permute(1, 0)).detach()

            # weight['st'] = torch.matmul(t_label_select,s_label_select.permute(1, 0).detach())

            # weight['st'] = torch.matmul(t_label_select, s_label_select.permute(1, 0))
            # print('weight',weight['st'][:num, :num].mean(),weight['st'][num:, num:].mean())
            # print('weight[]',weight['st'].shape,dist['st'].shape)
            dist['st']= torch.mul(weight['st'],dist['st'])

            # dist['tt'] = torch.mul(weight['tt'], dist['tt'])
        lossD = {}
        # lossD['ss']=0*-1/dist['ss'][:num,:num].mean()+1/(dist['ss'][:num,num:].mean()+0.001)+1/(dist['ss'][num:,:num].mean()+0.001)-0*1/dist['ss'][num:, num:].mean()
        # lossD['ss'] = dist['ss'][:num, :num].mean() - 0.1*dist['ss'][:num, num:].mean() - 0.1*dist['ss'][num:, :num].mean() + \
        #                   dist['ss'][num:, num:].mean()
        # # #
        # lossD['tt'] = dist['tt'][:num, :num].mean() - 0.1*dist['tt'][:num, num:].mean() - 0.1*dist['tt'][num:, :num].mean() + \
        #               dist['tt'][num:, num:].mean()
        #
        # lossD['st'] = dist['st'][:num, :num].mean() -0*dist['st'][:num, num:].mean() -\
        #               0*dist['st'][num:,:num].mean() + dist['st'][num:, num:].mean()
        # lossD['ss'] = (dist['ss'][:num, :num]/dist['ss'][:num, :num].max()).mean() - (dist['ss'][:num, num:]/dist['ss'][:num, num:].max()+0.1).mean() \
        #               -(dist['ss'][num:, :num]/dist['ss'][num:, :num].max()+0.1).mean() + (dist['ss'][num:, num:]/dist['ss'][num:, num:].max()).mean()
        # lossD['tt'] = (dist['tt'][:num, :num] / dist['tt'][:num, :num].max()).mean() +1/ (dist['tt'][:num, num:] / dist['tt'][:num, num:].max()+0.1).mean() \
        #               +1/ (dist['tt'][num:, :num] / dist['tt'][num:, :num].max()+0.1).mean() + (dist['tt'][num:, num:] / dist['tt'][num:, num:].max()).mean()

        lossD['st'] = dist['st'][:num, :num].mean() -  0.1*dist['st'][:num, num:].mean() - \
                       0.1*dist['st'][num:, :num].mean() + dist['st'][num:, num:].mean()

        # lossD['st'] = (dist['st'][:num, :num]/dist['st'][:num, :num].max()+1).mean() -0*(dist['st'][:num, num:]/dist['st'][:num, num:].max()+1).mean() \
        #               -0* (dist['st'][num:, :num]/dist['st'][num:, :num].max()+1).mean() + (dist['st'][num:, num:]/dist['st'][num:, num:].max()+1).mean()
        # print(dist['ss'][:num,:num].mean(),dist['ss'][num:, num:].mean())
        # lossD['st'] = dist['st'][:num, :num].mean() + 1/ (dist['st'][:num, num:].mean()+0.001) + \
        #                1/(dist['st'][num:, :num].mean()+0.001) + dist['st'][num:, num:].mean()
        # mmd = torch.mean(2*lossD['st']+(lossD['ss'] +lossD['tt']))
        # mmd = torch.mean(lossD['st']+lossD['ss'])
        mmd = torch.mean(lossD['st'])
        #
        #
        # print('dist[ss]', kernel_dist['ss'].shape, kernel_dist['ss'])
        # print('dist[tt]', kernel_dist['tt'].shape, kernel_dist['tt'])
        # print('dist[st]', kernel_dist['st'].shape, kernel_dist['st'])
        # mmd = torch.mean(kernel_dist['ss']) + torch.mean(kernel_dist['tt']) \
        #                 - 2.0 * torch.mean(kernel_dist['st'])
        return {'mmd': mmd}