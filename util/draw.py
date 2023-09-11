
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import pickle

def save_pickle(data, file_name):
	f = open(file_name, "wb")
	pickle.dump(data, f)
	f.close()
def load_pickle(file_name):
	f = open(file_name, "rb+")
	data = pickle.load(f)
	f.close()
	return data


class setFigure():
    # def __init__(self):
    def initialize_figure(self):
        self.metrics = {
            'nochange_acc': [],
            'change_acc': [],
            'prec': [],
            'rec': [],
            'f_meas': [],
            'Loss': [],
            'ce_loss': [],
            'total_acc': [],
            'focal_loss': [],
            'dice_loss':[],
            'Iou':[]

        }
        return self.metrics
    def set_figure(self,metric_dict, nochange_acc, change_acc, prec, rec, f_meas, Loss, ce_loss, total_acc,focal_loss,dice_loss,Iou):
        metric_dict['nochange_acc'].append(nochange_acc)
        metric_dict['change_acc'].append(change_acc)
        metric_dict['prec'].append(prec)
        metric_dict['rec'].append(rec)
        metric_dict['f_meas'].append(f_meas)
        metric_dict['Loss'].append(Loss)
        metric_dict['ce_loss'].append(ce_loss)
        metric_dict['total_acc'].append(total_acc)
        metric_dict['focal_loss'].append(focal_loss)
        metric_dict['dice_loss'].append(dice_loss)
        metric_dict['Iou'].append(Iou)


        return metric_dict

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

def get_parser_with_args(metadata_json='./utils/metadata_GZ.json'):
    parser = ag.ArgumentParser(description='Training change detection network')

    with open(metadata_json, 'r') as fin:
        metadata = json.load(fin)
        parser.set_defaults(**metadata)
        return parser, metadata
    return None
def plotFigure(figure_train_metrics,figure_test_metrics,num_epochs,name, time_now):
    t = np.linspace(1, num_epochs, num_epochs)
    e=num_epochs
    epoch_train_nochange_accuracy = figure_train_metrics['nochange_acc']
    epoch_train_change_accuracy = figure_train_metrics['change_acc']
    epoch_train_precision = figure_train_metrics['prec']
    epoch_train_recall = figure_train_metrics['rec']
    epoch_train_Fmeasure = figure_train_metrics['f_meas']
    epoch_train_Loss = figure_train_metrics['Loss']
    epoch_train_ce_loss = figure_train_metrics['ce_loss']
    epoch_train_accuracy = figure_train_metrics['total_acc']
    epoch_train_focal_loss = figure_train_metrics['focal_loss']
    epoch_train_dice_loss = figure_train_metrics['dice_loss']
    epoch_train_Iou = figure_train_metrics['Iou']

    epoch_test_nochange_accuracy = figure_test_metrics['nochange_acc']
    epoch_test_change_accuracy = figure_test_metrics['change_acc']
    epoch_test_precision = figure_test_metrics['prec']
    epoch_test_recall = figure_test_metrics['rec']
    epoch_test_Fmeasure = figure_test_metrics['f_meas']
    epoch_test_Loss= figure_test_metrics['Loss']
    epoch_test_ce_loss= figure_test_metrics['ce_loss']
    epoch_test_accuracy = figure_test_metrics['total_acc']

    epoch_test_focal_loss= figure_test_metrics['focal_loss']
    epoch_test_dice_loss= figure_test_metrics['dice_loss']
    epoch_test_Iou= figure_test_metrics['Iou']

    plt.figure(num=1)
    plt.clf()
    train_loss=[]
    test_loss=[]
    # for i in range(len(epoch_train_Loss[:e + 1])):
    #     train_loss.append(epoch_train_clf_loss[:e + 1][i]+epoch_train_marginLoss[:e + 1][i])
    #     test_loss.append(epoch_test_loss[:e + 1][i]+epoch_test_marginLoss[:e + 1][i])

    l1_1, = plt.plot(t[:e + 1], epoch_train_ce_loss[:e + 1], label='Train CE loss')
    l1_2, = plt.plot(t[:e + 1], epoch_test_ce_loss[:e + 1], label='Test CE loss')
    l1_3, = plt.plot(t[:e + 1], epoch_train_focal_loss[:e + 1], label='Train Focal loss')
    l1_4, = plt.plot(t[:e + 1], epoch_test_focal_loss[:e + 1], label='Test Focal loss')
    l1_5, = plt.plot(t[:e + 1], epoch_train_dice_loss[:e + 1], label='Train Dice loss')
    l1_6, = plt.plot(t[:e + 1], epoch_test_dice_loss[:e + 1], label='Test Dice loss')
    l1_7, = plt.plot(t[:e + 1], epoch_train_Loss, label='Train Total loss')
    l1_8, = plt.plot(t[:e + 1], epoch_test_Loss, label='Test Total loss')
    plt.legend(handles=[l1_1, l1_2, l1_3, l1_4, l1_5, l1_6,l1_7,l1_8])
    plt.grid()
    #         plt.gcf().gca().set_ylim(bottom = 0)
    plt.gcf().gca().set_xlim(left=0)
    plt.title('Loss')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=2)
    plt.clf()
    l2_1, = plt.plot(t[:e + 1], epoch_train_accuracy[:e + 1], label='Train accuracy')
    l2_2, = plt.plot(t[:e + 1], epoch_test_accuracy[:e + 1], label='Test accuracy')
    plt.legend(handles=[l2_1, l2_2])
    plt.grid()
    plt.gcf().gca().set_ylim(0, 1)
    plt.title('Accuracy')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=3)
    plt.clf()
    l3_1, = plt.plot(t[:e + 1], epoch_train_nochange_accuracy[:e + 1], label='Train accuracy: no change')
    l3_2, = plt.plot(t[:e + 1], epoch_train_change_accuracy[:e + 1], label='Train accuracy: change')
    l3_3, = plt.plot(t[:e + 1], epoch_test_nochange_accuracy[:e + 1], label='Test accuracy: no change')
    l3_4, = plt.plot(t[:e + 1], epoch_test_change_accuracy[:e + 1], label='Test accuracy: change')
    plt.legend(loc='best', handles=[l3_1, l3_2, l3_3, l3_4])
    plt.grid()
    plt.gcf().gca().set_ylim(0, 1)
    plt.title('Accuracy per class')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=4)
    plt.clf()
    l4_1, = plt.plot(t[:e + 1], epoch_train_precision[:e + 1], label='Train precision')
    l4_2, = plt.plot(t[:e + 1], epoch_train_recall[:e + 1], label='Train recall')
    l4_3, = plt.plot(t[:e + 1], epoch_train_Fmeasure[:e + 1], label='Train Dice/F1')
    l4_4, = plt.plot(t[:e + 1], epoch_test_precision[:e + 1], label='Test precision')
    l4_5, = plt.plot(t[:e + 1], epoch_test_recall[:e + 1], label='Test recall')
    l4_6, = plt.plot(t[:e + 1], epoch_test_Fmeasure[:e + 1], label='Test Dice/F1')
    l4_7, = plt.plot(t[:e + 1], epoch_train_Iou[:e + 1], label='Train Iou')
    l4_8, = plt.plot(t[:e + 1], epoch_test_Iou[:e + 1], label='Test Iou')
    plt.legend(loc='best', handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6,l4_7,l4_8])
    plt.grid()
    plt.gcf().gca().set_ylim(0, 1)
    #         plt.gcf().gca().set_ylim(bottom = 0)
    #         plt.gcf().gca().set_xlim(left = 0)
    plt.title('Precision, Recall , F-measure and Iou')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=5)
    plt.clf()
    l5_1, = plt.plot(t[:e + 1], epoch_train_ce_loss[:e + 1], label='Train CELoss')
    l5_2, = plt.plot(t[:e + 1], epoch_test_ce_loss[:e + 1], label='Test CELoss')
    plt.legend(loc='best', handles=[l5_1, l5_2])
    plt.grid()
    plt.gcf().gca().set_xlim(left=0)
    plt.title('CE Loss')
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.figure(num=6)
    plt.clf()
    l6_1, = plt.plot(t[:e + 1], epoch_train_Loss[:e + 1], label='Train Loss')
    l6_2, = plt.plot(t[:e + 1], epoch_test_Loss[:e + 1], label='Test Loss')

    plt.legend(loc='best', handles=[l6_1,l6_2])
    plt.grid()
    plt.gcf().gca().set_xlim(left=0)
    plt.title('Total Loss')
    display.clear_output(wait=True)
    display.display(plt.gcf())


    save = True
    if save:
        plt.figure(num=1)
        # plt.savefig('./log/%s/%s/%s-01-loss.png' % (name, time_now, name))
        plt.savefig('../01-loss.png' )

        plt.figure(num=2)
        # plt.savefig('./log/%s/%s/%s-02-accuracy.png' % (name, time_now, name))
        plt.savefig('../02-accuracy.png' )


        plt.figure(num=3)
        # plt.savefig('./log/%s/%s/%s-03-accuracy_per_class.png' % (name, time_now, name))
        plt.savefig('../03-accuracy_per_class.png')

        plt.figure(num=4)
        # plt.savefig('./log/%s/%s/%s-04-prec_rec_fmeas.png' % (name, time_now, name))
        plt.savefig('../04-prec_rec_fmeas.png' )
        plt.figure(num=5)
        # plt.savefig('./log/%s/%s/%s-05-var.png' % (name, time_now, name))
        plt.savefig('../05-CELoss.png' )

        plt.figure(num=6)
        # plt.savefig('./log/%s/%s/%s-06-trans.png' % (name, time_now, name))
        plt.savefig('../06-TLoss.png' )

if __name__ == '__main__':
    figure_train_metrics = load_pickle("../fig_train.pkl")
    figure_test_metrics = load_pickle("../fig_test.pkl")

    # figure_train_metrics=np.load('fig_train.npy')
    # figure_test_metrics=np.load('fig_test.npy')
    num_epochs=len(figure_train_metrics['nochange_acc'])
    plotFigure(figure_train_metrics, figure_test_metrics, num_epochs)
