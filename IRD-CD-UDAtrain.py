import random
from data.data_loader import CreateDataLoader
from util.train_util import *

from option.train_options import TrainOptions
from util.visualizer import Visualizer
from util.metric_tool import ConfuseMatrixMeter
import math
from tqdm import tqdm
from util.drawTool import initialize_weights,save_pickle,load_pickle
from util.drawTool import setFigure,add_weight_decay,plotFigure,MakeRecordFloder,confuseMatrix,setFigureDA
from torch.autograd import Variable
from option.config import cfg
from modelDA import utils as model_utils
from modelDA import HLCDNetSBN2M2
from util.metrics_DA import CORAL,MMD_loss,SelecFeat
from modelDA.mmd import MMD

def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0
def compute_paired_dist(A, B):

    bs_A = A.size(0)
    bs_T = B.size(0)
    feat_len = A.size(1)

    A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
    B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
    # print(A_expand.shape)#[2400, 2400, 32]

    dist=F.cosine_similarity(A_expand,B_expand,dim=2)
    return dist
if __name__ == '__main__':
    gpu_id = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    opt = TrainOptions().parse(save=False,gpu=gpu_id)
    opt.num_epochs=10
    opt.batch_size=5
    opt.use_ce_loss = True
    opt.use_hybrid_loss = False
    opt.use_UnchgInCenterLoss = False
    opt.use_UnchgInCenterLossNew = False
    opt.use_UnchgNoCenterLoss = False
    opt.num_threads=opt.num_threads
    opt.num_decay_epochs=0
    opt.dset='CD_DA_building'
    opt.model_type='HLCDNetSBN2'
    cfg.DA.NUM_DOMAINS_BN = 2
    ttest=False
    name = opt.dset + 'DAmmdnew2DAComapre11/' + opt.model_type + '_load_CE_sample_coral_W_STp_kernal_G-L'  # '_CE_IN_PreRest_noweightC_PCE',noload!
    lrWeight = [.5, .5, 1, 1]
    opt.LChannel = True
    opt.dataroot = opt.dataroot + '/' + opt.dset
    opt.s = 0
    opt.t = 1
    cfg.DA.S = opt.s
    cfg.DA.T = opt.t
    if opt.dset == 'CD_DA_building':
        # cfg.TRAINLOG.DATA_NAMES = ['SYSU_CD', 'LEVIR_CDPatch', 'GZ_CDPatch']
        cfg.TRAINLOG.DATA_NAMES = ['LEVIR_CDPatch', 'GZ_CDPatch']

    cfg.TRAINLOG.EXCEL_LOGSheet = ['wsT-' + cfg.TRAINLOG.DATA_NAMES[0][0], 'wsT-' + cfg.TRAINLOG.DATA_NAMES[1][0],
                                   'wsTr', 'wsVal']
    saveroot = './log/CD_DA_buildingNew/M2CE_CE/20230911-14_44_LEVIR_CDPatch-GZ_CDPatch'  # /savemodel/
    save_path = saveroot + '/savemodel/_2_acc-0.9059_chgAcc-0.4201_unchgAcc-0.9588.pth'
    opt.load_pretrain = False
    opt.load_pretrainDA = True
    opt.load_pretrain_low=False

    resnet_pretrained = True
    print('\n########## Recording File Initialization#################')
    SEED = opt.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg.TRAINLOG.STARTTIME = time.strftime("%Y%m%d-%H_%M", time.localtime())
    time_now = time.strftime("%Y%m%d-%H_%M_"+cfg.TRAINLOG.DATA_NAMES[opt.s], time.localtime())
    filename_main = os.path.basename(__file__)
    start_epoch, epoch_iter=MakeRecordFloder(name, time_now, opt,filename_main,opt.load_pretrain,saveroot)
    train_metrics = setFigureDA()
    val_metrics = setFigureDA()
    figure_train_metrics = train_metrics.initialize_figure()
    figure_val_metrics = val_metrics.initialize_figure()

    print('\n########## Load the Source Dataset #################')

    opt.phase = 'train'
    train_loader = CreateDataLoader(opt)
    train_data = train_loader.load_data()
    train_data_len = len(train_data)
    train_size = len(train_loader)
    print("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase,cfg.TRAINLOG.DATA_NAMES[opt.s],train_size))
    cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
          (opt.phase,cfg.TRAINLOG.DATA_NAMES[opt.s],train_size) + '\n')


    print('\n########## Load the Target Dataset #################')

    t_loaderDict = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
        # if i == opt.s:
            opt.t = i
            opt.phase = 'target_train'
            t_loader = CreateDataLoader(opt)
            t_loaderDict.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            t_data = t_loader.load_data()
            t_data_len = len(t_data)
            t_size = len(t_loader)
            print("[%s] dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size))
            cfg.TRAINLOG.LOGTXT.write("[%s] dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size) + '\n')
    print('t_loaderDict',t_loaderDict)

    t_loaderDicttest = {}
    for i in range(len(cfg.TRAINLOG.DATA_NAMES)):
        if i != opt.s:
        # if i == opt.s:
            opt.t = i
            opt.phase = 'target'
            t_loader = CreateDataLoader(opt)
            t_loaderDicttest.update({cfg.TRAINLOG.DATA_NAMES[i]: t_loader})
            t_data = t_loader.load_data()
            t_data_len = len(t_data)
            t_size = len(t_loader)
            print("[%s] test dataset [%s] was created successfully! Num= %d" %
                  (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size))
            cfg.TRAINLOG.LOGTXT.write("[%s] test dataset [%s] was created successfully! Num= %d" %
                                      (opt.phase, cfg.TRAINLOG.DATA_NAMES[opt.t], t_size) + '\n')
    print('t_loaderDicttest',t_loaderDicttest)
    tool = CDModelutil(opt)
    # initialize model
    model_state_dict = None
    # resume_dict = None
    ######################################################################################################
    if opt.load_pretrain:
        # saveroot='./log/CD_DA_building/HLCDNet2_a/20230316-23_34_GZ_CDPatch'
        # save_path=saveroot+'/savemodel/_101_acc-0.9808_chgAcc-0.9316_unchgAcc-0.9849.pth'
        figure_train_metrics = load_pickle(saveroot+"/fig_train.pkl")
        figure_val_metrics = load_pickle(saveroot+"/fig_val.pkl")
        start_epoch = len(figure_train_metrics['nochange_acc'])+1
        print('start_epochstart_epochstart_epoch',start_epoch,'end:',opt.num_epochs + opt.num_decay_epochs + 1)
        model_state_dict,bn_domain_map,optimizer_state_dict=tool.load_dackpt(save_path)
        opt.num_epochs=opt.num_epochs+start_epoch
        cfg.DA.BN_DOMAIN_MAP = bn_domain_map
        print('ckpt load BN:',bn_domain_map)
    elif opt.load_pretrain_low:
        print('load_pretrain_low')
        model_state_dict, bn_domain_map, optimizer_state_dict = tool.load_lowPretrain(save_path)
        model_num = 2
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
    elif opt.load_pretrainDA:
        model_state_dict, bn_domain_map, optimizer_state_dict = tool.load_lowPretrain(save_path)
        # model_num = 2
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
    else:
        cfg.DA.BN_DOMAIN_MAP = {cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]: 0, cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]: 1}
    print('cfg.DA.BN_DOMAIN_MAP',cfg.DA.BN_DOMAIN_MAP)
    print('\n########## Build the Molde #################')
    opt.phase = 'train'
    netAL = HLCDNetSBN2M2.LowEx(num_domains_bn=cfg.DA.NUM_DOMAINS_BN)
    netAH=HLCDNetSBN2M2.BaseNet(in_dim=3, out_dim=2, pretrained=resnet_pretrained, num_domains_bn=cfg.DA.NUM_DOMAINS_BN)
    netB=HLCDNetSBN2M2.DeepDeconv(num_domains_bn=cfg.DA.NUM_DOMAINS_BN)
    netC=HLCDNetSBN2M2.CD_classifer(num_domains_bn=cfg.DA.NUM_DOMAINS_BN)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('DEVICE:', DEVICE)
    for i, net in enumerate([netAL, netAH, netB, netC]):
        net.to(DEVICE)
        if opt.load_pretrain_low:
            if model_state_dict is not None  and i < model_num:
                model_utils.init_weights(net, model_state_dict[i], bn_domain_map, BN2BNDomain=False)
            if opt.load_pretrain_low and i == 1:
                for p in net.parameters():
                    p.requires_grad = False  # param.requires_grad=False对BN不起任何作用
        elif opt.load_pretrainDA or opt.load_pretrain:
            if model_state_dict is not None:
                model_utils.init_weights(net, model_state_dict[i], bn_domain_map, BN2BNDomain=False)

        else:
            continue
    print('\n########## Load the Optimizer #################')

    param_group = []
    for i,net in enumerate([netAL,netAH,netB,netC]):
        for k, v in net.named_parameters():
            param_group += [{'params': v, 'lr': opt.lr * lrWeight[i]}]

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_group, lr=opt.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_group, lr=opt.lr,
                                     betas=(0.5, 0.999))
    else:
        raise NotImplemented(opt.optimizer)
    if opt.load_pretrain:
        optimizer.load_state_dict(optimizer_state_dict)
    print('optimizer:', opt.optimizer)
    cfg.TRAINLOG.LOGTXT.write('optimizer: ' + opt.optimizer + '\n')

    visualizer = Visualizer(opt)
    tmp = 1
    running_metric = ConfuseMatrixMeter(n_class=2)
    TRAIN_ACC = np.array([], np.float32)
    VAL_ACC = np.array([], np.float32)
    best_val_acc = 0.0
    best_epoch = 0
    # ws = cfg.TRAINLOG.EXCEL_LOG['Sheet']
    coral_loss = CORAL()
    mmd_loss = MMD_loss()
    mmd = MMD(num_layers=1, kernel_num=[3],
                   kernel_mul=[2], joint=False,device=DEVICE)
    ratio=0
    ones,zeros=None,None
    # mmds = MMDS(num_layers=1, kernel_num=[3],
    #             kernel_mul=[2], joint=False, device=DEVICE)
    for epoch in range(start_epoch, opt.num_epochs + opt.num_decay_epochs + 1):


        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % train_size
        running_metric.clear()
        opt.phase = 'train'
        train_data=train_loader.load_data()
        iter_source = iter(train_data)
        len_source_loader = train_data_len
        Tt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        len_target_loader = len(Tt_data)
        # tbar = tqdm(range(t_data_len))
        iter_target = iter(Tt_data)

        print('len_source_loader', len_source_loader, len_target_loader)
        tbar = tqdm(range(len_target_loader - 1))
        train_data_len = len_target_loader
        ce_lossT = 0
        FeatLossT = 0
        DAlow_lossT = 0
        DAEntropyT=0
        DATlossT=0
        LossT = 0
        CdistT=0
        # KMMD = MMD(num_layers=2, kernel_num=(5,5),kernel_mul=(2,2), joint=False)
        mNum=0
        selectF = SelecFeat()
        for net in [netAL, netAH, netB, netC]:
            # net.zero_grad()
            net.train()
            net.set_bn_domain(cfg.DA.BN_DOMAIN_MAP[cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]])
        for i in tbar:
            try:
                Sdata = next(iter_source)
            except:
                iter_source = iter(train_data)
            try:
                Tdata = next(iter_target)
            except:
                iter_target = iter(t_loader)
            epoch_iter += opt.batch_size
            a=random.randint(1,2)

            inputH, L_AC_outS = netAL(Variable(Sdata['t1_img']).to(DEVICE), Variable(Sdata['t2_img']).to(DEVICE))
            FusionFeatS, featHS = netAH(inputH[0], inputH[1], inputH[2])
            defeat3S = netB(FusionFeatS, featHS)
            preS = netC(defeat3S)
            cd_predS = [preS, L_AC_outS, defeat3S]
            labelS = Variable(Sdata['label']).to(DEVICE)
            # calculate final loss scalar
            if opt.use_ce_loss:
                CELoss = tool.loss(cd_predS[0], labelS.long())
                loss = CELoss
                FeatLoss = 0
                loss_tr = CELoss
                CELoss = CELoss.item()
                # LossT = LossT + loss_tr
                ce_lossT = ce_lossT + CELoss
                FeatLossT = FeatLossT + FeatLoss
                # diceLossT = diceLossT + DiceLoss

            else:
                raise NotImplementedError
            inputH, L_AC_outT = netAL(Variable(Tdata['t1_img']).to(DEVICE), Variable(Tdata['t2_img']).to(DEVICE))
            FusionFeatT, featHT = netAH(inputH[0], inputH[1], inputH[2])
            defeat3T = netB(FusionFeatT, featHT)
            preT = netC(defeat3T)
            labelT=Variable(Tdata['label']).to(DEVICE)
            labelTori=labelT
            cd_predT = [preT, L_AC_outT, defeat3T]
            preTS=F.softmax(preT,dim=1)
            cd_predTo = torch.argmax(preTS.detach(), dim=1)
            # #####Center
            DA=True
            if DA:
                preT = preT.reshape(preT.shape[0], preT.shape[1], -1)
                labelT=labelT.reshape(labelT.shape[0],-1)
                # reshapePreTSoftmax = F.softmax(preT,dim=1)
                preT = F.log_softmax(preT, dim=1)
                pt=0.9-epoch/50
                # pt=0
                source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select, s_label_select, t_label_select, _, softLog = \
                    selectF.select_featureST(defeat3S, labelS, defeat3T, cd_predTo.squeeze(0),
                                             preTS, [preS, preT], p=pt, device=DEVICE)
                feats_toalign_S = [torch.cat([source_unchg_flatten_select, source_chg_flatten_select], dim=0).to(DEVICE)]
                feats_toalign_T = [torch.cat([target_unchg_flatten_select, target_chg_flatten_select], dim=0).to(DEVICE)]
                Schg, Sunchg = selectF.select_featureS(source=defeat3S, s_label=labelS)
                # preTen = cd_predT[0].reshape(-1, cd_predT[0].shape[1])
                # preTen = F.softmax(preTen,dim=1)
                # ent_loss = torch.mean(tool.entropy(preTen,1))  #
                # [entropyunchg, entropychg] = (tool.entropy(preTen, 1))  #
                ent_loss=torch.tensor([0]).to(DEVICE)
                SchgC = Schg.mean(dim=0).detach()
                SunchgC = Sunchg.mean(dim=0).detach()
                if feats_toalign_S[0].shape[0]>selectF.chgthreshold*2*0.4 and feats_toalign_T[0].shape[0]>selectF.unchgthreshold*2*0.4 and selectF.chgNum==selectF.unchgNum:

                    mmd_loss = mmd.forward(feats_toalign_S, feats_toalign_T,s_label_select,t_label_select)['mmd']
                    coral_loss_value = coral_loss.CORAL(cd_predS[1], cd_predT[1])

                    pseudoCE = mmd_loss + coral_loss_value+a*ent_loss
                    Score = {'CEL': CELoss, 'DAlow': mmd_loss.item(), 'FeatL': (ent_loss).item(), 'Cd': selectF.unchgNum/100+1,
                             'c':mmd.maxc.item(),'u':mmd.maxu.item()}
                    mNum=mNum+1
                else:
                    pe=0.1
                    source_chg_flatten_select, source_unchg_flatten_select, target_chg_flatten_select, target_unchg_flatten_select, s_label_select, _, t_label_select, softLog = \
                        selectF.select_featureST(defeat3S, labelS, defeat3T, cd_predTo.squeeze(0),
                                                 preTS, [preS, preT], p=pt, pe=pe, device=DEVICE)

                    feats_toalign_S = [torch.cat([source_unchg_flatten_select, source_chg_flatten_select], dim=0).to(DEVICE)]
                    feats_toalign_T = [torch.cat([target_unchg_flatten_select, target_chg_flatten_select], dim=0).to(DEVICE)]
                    if feats_toalign_S[0].shape[0] > 1 and feats_toalign_T[0].shape[0] > 1 and selectF.chgNum == selectF.unchgNum:
                        mmd_loss = mmd.forward(feats_toalign_S, feats_toalign_T, s_label_select, t_label_select)['mmd']
                        coral_loss_value = coral_loss.CORAL(cd_predS[1], cd_predT[1])
                        pseudoCE = mmd_loss + coral_loss_value+a*ent_loss
                        Score = {'CEL': CELoss, 'DAlow': mmd_loss.item(), 'FeatL': (coral_loss_value+ent_loss).item(), 'Cd': selectF.unchgNum / 100}
                    else:

                        mmd_loss=torch.Tensor(0).to(DEVICE)
                        coral_loss_value = coral_loss.CORAL(cd_predS[1], cd_predT[1])#noweight
                        pseudoCE = 0.1*(coral_loss_value+a*ent_loss)
                        Score = {'CEL': CELoss, 'DAlow': pseudoCE.item(), 'FeatL': coral_loss_value.item(), 'Cd': 0}
                labelTNp = labelT.unsqueeze(1).cpu().numpy()
                pseudoNp = cd_predTo.squeeze(0).cpu().numpy()
                CM = running_metric.confuseM(pr=pseudoNp,presoft=cd_predTo.cpu().numpy(), gt=labelTNp)

            train_target = Sdata['label'].detach()
            cd_predSo = torch.argmax(cd_predS[0].detach(), dim=1)
            current_score = running_metric.update_cm(pr=cd_predSo.cpu().numpy(), gt=train_target.cpu().numpy())
            Score.update(current_score)
            if DA : Score.update(CM)
            trainMessage = visualizer.print_current_scores(opt.phase, epoch, i, train_data_len, Score)
            tbar.set_description(trainMessage)
            ############### Backward Pass ####################

            optimizer.zero_grad()
            (loss+pseudoCE).backward()######################################
            LossT=LossT+loss_tr.item()
            DAlow_lossT = DAlow_lossT
            DAEntropyT = DAEntropyT
            DATlossT=DATlossT+pseudoCE.item()
            CdistT=CdistT
            # loss.backward()
            optimizer.step()
            if i > 5 and ttest:
                break
        lossAvg = LossT / i
        ce_lossAvg = ce_lossT / i
        FeatLossAvg = FeatLossT / i
        DAlow_lossAvg = DAlow_lossT / i
        DAEntropyAvg=DAEntropyT/i
        DATlossTAvg=DATlossT/i
        CdistTAvg=CdistT/i
        ratio = mNum / i
        print('ratioratio:', ratio)
        # print('TTT:',messageT)
        train_scores = running_metric.get_scores()
        IterScore = {'Loss':lossAvg,'CE': ce_lossAvg,
                     'DAT':DATlossTAvg,'DALow': DAlow_lossAvg, 'FeatL': FeatLossAvg,'DAEnt':DAEntropyT,"Cdist":CdistTAvg}
        IterScore.update(train_scores)
        message=visualizer.print_scores(opt.phase, epoch, IterScore)
        messageT,core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message+ messageT+'\n')

        exel_out = opt.phase, epoch, ce_lossAvg, DAlow_lossAvg,DAEntropyAvg, DATlossTAvg,FeatLossAvg, IterScore['acc'], IterScore['unchgAcc'], \
                   IterScore['chgAcc'], IterScore['recall_1'], \
                   IterScore['F1_1'], IterScore['mf1'],IterScore['miou'], IterScore['precision_1'], \
                   str(IterScore['tn']), str(IterScore['tp']), str(IterScore['fn']), str(IterScore['fp']),\
                   core_dictT['accT'],core_dictT['unchgT'],core_dictT['chgT'],core_dictT['mF1T']
        # ws.append(exel_out)
        cfg.TRAINLOG.EXCEL_LOG['wsTr'].append(exel_out)

        figure_train_metrics = train_metrics.set_figure(figure_train_metrics, IterScore['unchgAcc'],
                                                        IterScore['chgAcc'],
                                                        IterScore['precision_1'], IterScore['recall_1'],
                                                        train_scores['F1_1'], lossAvg, ce_lossAvg, train_scores['acc'],
                                                        DAlow_lossAvg,DAEntropyAvg,DATlossTAvg, FeatLossAvg, train_scores['miou'])


        ####################################val
        running_metric.clear()
        opt.phase = 'val'

        # TTt_loader = t_loaderDicttest[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        # # TTt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        #
        # TTt_data = TTt_loader.load_data()
        Tt_loader = t_loaderDict[cfg.TRAINLOG.DATA_NAMES[cfg.DA.T]]
        Tt_data = Tt_loader.load_data()
        targett_loader_len = len(Tt_data)
        tbar = tqdm(range(targett_loader_len))
        # iter_target = iter(TTt_data)
        iter_target = iter(Tt_data)
        ce_lossT = 0
        FeatLossT = 0
        LossT = 0
        with torch.no_grad():
            for net in [netAL,netAH, netB, netC]:
                net.set_bn_domain(cfg.DA.BN_DOMAIN_MAP[cfg.TRAINLOG.DATA_NAMES[cfg.DA.S]])
                net.eval()
            for i in tbar:
                data_val = next(iter_target)
                # data_T1_val=Variable(data_val['t1_img']).to(DEVICE)
                # data_T2_val=Variable(data_val['t2_img']).to(DEVICE)
                inputHval, L_AC_outVal = netAL(Variable(data_val['t1_img']).to(DEVICE), Variable(data_val['t2_img']).to(DEVICE))
                FusionFeatVal, featHVal = netAH(inputHval[0], inputHval[1], inputHval[2])
                defeat3Val = netB(FusionFeatVal, featHVal)
                preVal = netC(defeat3Val)
                cd_val_pred = [preVal, L_AC_outVal, defeat3Val]
                label=Variable(data_val['label']).to(DEVICE)

                if opt.use_ce_loss:
                    CELoss = tool.loss(cd_val_pred[0], label.long())
                    loss = CELoss
                    FeatLoss = 0
                    CELoss = CELoss.item()
                    loss_tr = CELoss
                    LossT = LossT + loss_tr
                    ce_lossT = ce_lossT + CELoss
                    FeatLossT = FeatLossT + FeatLoss
                else:
                    raise NotImplementedError
                # update metric
                Score = {'CEL': CELoss, 'DALoss': 0, 'FeatLoss': FeatLoss}
                val_target = data_val['label'].detach()
                val_pred = torch.argmax(cd_val_pred[0].detach(), dim=1)
                current_score = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())#更新
                Score.update(current_score)
                # Score.update(CM)

                valMessage=visualizer.print_current_scores(opt.phase,epoch,i,targett_loader_len,Score)
                tbar.set_description(valMessage)
                if i > 50 and ttest:
                    break
        val_scores = running_metric.get_scores()
        lossAvg = LossT / i
        ce_lossAvg = ce_lossT / i
        FeatLossAvg = FeatLossT / i
        IterValScore = {'Loss':lossAvg,'CE': ce_lossAvg, 'DALoss': 0, 'FeatLoss': FeatLossAvg}
        IterValScore.update(val_scores)
        message=visualizer.print_scores(opt.phase+'-'+cfg.TRAINLOG.DATA_NAMES[cfg.DA.T][0], epoch, IterValScore)
        # messageT, core_dictT = running_metric.get_scoresT()
        cfg.TRAINLOG.LOGTXT.write(message+ '\n')
        # print('val:',messageT)
        exel_out = opt.phase + '-' + cfg.TRAINLOG.DATA_NAMES[cfg.DA.T][0], epoch, ce_lossAvg, 0,0,0, FeatLossAvg, \
                   val_scores['acc'], \
                   val_scores['unchgAcc'], val_scores['chgAcc'], val_scores['recall_1'], \
                   val_scores['F1_1'],val_scores['mf1'], val_scores['miou'], val_scores['precision_1'], \
                   str(val_scores['tn']), str(val_scores['tp']), str(val_scores['fn']), str(val_scores['fp']),\
                   core_dictT['accT'],core_dictT['unchgT'],core_dictT['chgT'],core_dictT['mF1T']

        cfg.TRAINLOG.EXCEL_LOG['wsVal'].append(exel_out)

        figure_val_metrics = val_metrics.set_figure(figure_val_metrics, val_scores['unchgAcc'],
                                                    val_scores['chgAcc'],
                                                    val_scores['precision_1'], val_scores['recall_1'],
                                                    val_scores['F1_1'], lossAvg, ce_lossAvg,
                                                    val_scores['acc'],
                                                    0, 0,0,FeatLossAvg, val_scores['miou'])

        val_epoch_acc = val_scores['acc']
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch

        save_str = './log/%s/%s/savemodel/_%d_acc-%.4f_chgAcc-%.4f_unchgAcc-%.4f.pth' \
                   % (name, time_now, epoch + 1, val_scores['acc'], val_scores['chgAcc'], val_scores['unchgAcc'])

        tool.save_ckptDApre(iters=epoch,network=[netAL,netAH,netB,netC],bn_domain_map=cfg.DA.BN_DOMAIN_MAP,optimizer=optimizer,save_str=save_str)
        save_pickle(figure_train_metrics, "./log/%s/%s/fig_train.pkl" % (name, time_now))
        save_pickle(figure_val_metrics, "./log/%s/%s/fig_val.pkl" % (name, time_now))
        # end of epoch
        # print(opt.num_epochs,opt.num_decay_epochs)
        iter_end_time = time.time()
        # print('End of epoch %d / %d \t Time Taken: %d sec \t best acc: %.5f (at epoch: %d) ' %
        #     (epoch, opt.num_epochs + opt.num_decay_epochs, time.time() - epoch_start_time, best_val_acc, best_epoch))
        # np.savetxt(cfg.TRAINLOG.ITER_PATH, (epoch + 1, 0), delimiter=',', fmt='%d')
        cfg.TRAINLOG.EXCEL_LOG.save('./log/%s/%s/log.xlsx' % (name, time_now))


    print('================ Training Completed (%s) ================\n' % time.strftime("%c"))
    cfg.TRAINLOG.LOGTXT.write('\n================ Training Completed (%s) ================\n' % time.strftime("%c"))
    plotFigure(figure_train_metrics, figure_val_metrics, opt.num_epochs + opt.num_decay_epochs, name,opt.model_type, time_now)
    time_end = time.strftime("%Y%m%d-%H_%M", time.localtime())

    print('Training Start Time:', cfg.TRAINLOG.STARTTIME, '  Training Completion Time:', time_end, '  Total Epoch Num:', epoch)
    print('saved path:', './log/{}/{}'.format(name, time_now))

    cfg.TRAINLOG.LOGTXT.write('Training Start Time:'+ cfg.TRAINLOG.STARTTIME+ '  Training Completion Time:'+ time_end+ 'Total Epoch Num:'+ str(epoch) + '\n')





