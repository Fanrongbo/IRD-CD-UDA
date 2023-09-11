import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.networks import *
from model.loss import cross_entropy, Hybrid
from util import util
import time
import torch.nn.init as init


class CDModelutil():
    def __init__(self,opt):
        self.loss_filter = self.init_loss_filter(opt.use_ce_loss, opt.use_hybrid_loss)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if opt.use_ce_loss:
            self.loss = cross_entropy
        elif opt.use_hybrid_loss:
            self.loss = Hybrid(gamma=opt.gamma).to(self.device)
        else:
            raise NotImplementedError
        self.loss_names = self.loss_filter('CE', 'Focal', 'Dice')
    def load_ckpt(self, network, optimizer, which_epoch):
        save_filename = 'epoch_%s.pth' % which_epoch
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            raise ('%s must exist!' % (save_filename))
        else:
            checkpoint = torch.load(save_path,
                                    map_location=self.device)
            network.load_state_dict(checkpoint['network'], False)
    def init_loss_filter(self, use_ce_loss, use_hybrid_loss):
        flags = (use_ce_loss, use_hybrid_loss, use_hybrid_loss)#损失函数

        def loss_filter(ce, focal, dice):
            return [l for (l, f) in zip((ce, focal, dice), flags) if f]

        return loss_filter
    def init_method(self,net, init_type='normal'):
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
class CDModel(nn.Module):
    def name(self):
         return 'CDModel'


    def init_loss_filter(self, use_ce_loss, use_hybrid_loss):
        flags = (use_ce_loss, use_hybrid_loss, use_hybrid_loss)#损失函数

        def loss_filter(ce, focal, dice):
            return [l for (l, f) in zip((ce, focal, dice), flags) if f]

        return loss_filter


    def initialize(self, opt):
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() and len(opt.gpu_ids) > 0
                                   else "cpu")
        self.Tensor = torch.cuda.FloatTensor if self.device else torch.Tensor
        self.num_class = opt.num_class
        self.opt = opt
        self.old_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name, 'trained_models')
        util.mkdirs([self.save_dir])
        print('opt.gpu_idsopt.gpu_ids',opt.gpu_ids)
        # define model
        self.model = define_model(model_type=opt.model_type, resnet=opt.resnet, init_type=opt.init_type,
                                  initialize=opt.initialize, gpu_ids=opt.gpu_ids)

        # define optimizers
        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr,
                                       momentum=0.9,
                                       weight_decay=5e-4)
        elif opt.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr,
                                        betas=(0.5, 0.999))
        else:
            raise NotImplemented(opt.optimizer)


            # load models
        if opt.load_pretrain:
            self.load_ckpt(self.model, self.optimizer, opt.which_epoch)

        print('---------- Networks initialized -------------')

        # define loss functions
        self.loss_filter = self.init_loss_filter(opt.use_ce_loss, opt.use_hybrid_loss)
        if opt.use_ce_loss:
            self.loss = cross_entropy
        elif opt.use_hybrid_loss:
            self.loss = Hybrid(gamma=opt.gamma).to(self.device)
        else:
            raise NotImplementedError
        self.loss_names = self.loss_filter('CE', 'Focal', 'Dice')

    def forward(self, t1_input, t2_input, label,val=False):
        if val:
            self.model.eval()
            with torch.no_grad():
                pred = self.model.forward(t1_input, t2_input)
                print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
                time.sleep(4)
                print('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
                # print('nonoo')
                if self.opt.use_ce_loss:
                    ce_loss = self.loss(pred[0], label.long())
                    focal_loss = 0
                    dice_loss = 0
                elif self.opt.use_hybrid_loss:
                    ce_loss = 0
                    focal_loss, dice_loss = self.loss(pred[0], label.long())
                else:
                    raise NotImplementedError
        else:
            pred = self.model(t1_input, t2_input)
            # print('aa')
            # time.sleep(5)
            # print('bb')
            if self.opt.use_ce_loss:
                ce_loss = self.loss(pred[0], label.long())
                focal_loss = 0
                dice_loss = 0
            elif self.opt.use_hybrid_loss:
                ce_loss = 0
                focal_loss, dice_loss = self.loss(pred[0], label.long())
            else:
                raise NotImplementedError

        return [self.loss_filter(ce_loss, focal_loss, dice_loss), pred]


    def inference(self, t1_input, t2_input):
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                pred = self.model.forward(t1_input, t2_input)
                print('ddddddddddddddddddddddssssssssss')
        else:
            pred = self.model.forward(t1_input, t2_input)
            print('dddddssssssdddddddddddddddddssdddddddddddddddddssssssssss')

        return pred


    def load_ckpt(self, network, optimizer, which_epoch):
        save_filename = 'epoch_%s.pth' % which_epoch
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            raise ('%s must exist!' % (save_filename))
        else:
            checkpoint = torch.load(save_path,
                                    map_location=self.device)
            network.load_state_dict(checkpoint['network'], False)
            # optimizer.load_state_dict(checkpoint['optimizer'], False)


    def save_ckpt(self, network, optimizer, save_str):
        # save_filename = 'epoch_%s.pth' % which_epoch
        # save_path = os.path.join(self.save_dir, save_filename)
        save_path=save_str
        torch.save({
                    'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
            save_path)
        if torch.cuda.is_available():
            network.cuda()


    def save(self, model,optimizer,save_str):
        self.save_ckpt(model, optimizer, save_str)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.num_decay_epochs
        lr = self.old_lr - lrd

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate to %f' % lr)
        self.old_lr = lr


class InferenceModel(CDModel):
    def forward(self, t1_input, t2_input):
        return self.inference(t1_input, t2_input)


def create_model(opt):
    model = CDModel()
    model.initialize(opt)

    print("model [%s] was created" % (model.name()))

    if len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
