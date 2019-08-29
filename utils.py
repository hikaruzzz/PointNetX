# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def show_example(x, y, x_reconstruction, y_pred,save_dir, figname):
    x = x.squeeze().cpu().data.numpy()
    x = x.permute(0,2,1)
    y = y.cpu().data.numpy()
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)
    y_pred = y_pred.cpu().data.numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.savefig(save_dir + figname + '.png')

def save_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, path,modelnet='checkpoint'):
    savepath  = path + '/%s-%f-%04d.pth' % (modelnet,test_accuracy, epoch)
    state = {
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)

def test(model, loader):
    mean_correct = []
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    return np.mean(mean_correct)

def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_partseg(model, loader, catdict, num_classes = 50,forpointnet2=False):
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    iou_tabel = np.zeros((len(catdict),3))
    iou_list = []
    metrics = defaultdict(lambda:list())
    hist_acc = []
    # mean_correct = []
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point,_= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
        if forpointnet2:
            seg_pred = model(points, norm_plt, to_categorical(label, 16))
        else:
            labels_pred, seg_pred, _  = model(points,to_categorical(label,16))
            # labels_pred_choice = labels_pred.data.max(1)[1]
            # labels_correct = labels_pred_choice.eq(label.long().data).cpu().sum()
            # mean_correct.append(labels_correct.item() / float(points.size()[0]))
        # print(pred.size())
        iou_tabel, iou = compute_cat_iou(seg_pred,target,iou_tabel)
        iou_list+=iou
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['inctance_avg_iou'] = np.mean(iou_list)
    # metrics['label_accuracy'] = np.mean(mean_correct)
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    metrics['class_avg_iou'] = np.mean(cat_iou)

    return metrics, hist_acc, cat_iou

def test_semseg(model, loader, catdict, num_classes, pointnet2=False):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    #for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
    for batch_id, (points, target) in enumerate(loader):
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        else:
            pass
        if pointnet2:
            pred = model(points[:, :3, :], points[:, 3:, :])
        else:
            pred, _ = model(points)
        # print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean

def plot_loss_curve(history,n_points_avg,n_points_plot,save_dir):
    curve = np.asarray(history['loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-g')

    curve = np.asarray(history['margin_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-b')

    curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-r')

    plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
    plt.savefig(save_dir + '/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M')) + '_total_result.png')
    plt.close()

def plot_acc_curve(total_train_acc,total_test_acc,save_dir):
    plt.plot(total_train_acc, '-b',label = 'train_acc')
    plt.plot(total_test_acc, '-r',label = 'test_acc')
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Accuracy of training and test')
    plt.savefig(save_dir +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'_total_acc.png')
    plt.close()

def show_point_cloud(tuple,seg_label=[],title=None):
    import matplotlib.pyplot as plt
    if seg_label == []:
        x = [x[0] for x in tuple]
        y = [y[1] for y in tuple]
        z = [z[2] for z in tuple]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        category = list(np.unique(seg_label))
        color = ['b','r','g','y','w','b','p']
        ax = plt.subplot(111, projection='3d')
        for categ_index in range(len(category)):
            tuple_seg = tuple[seg_label == category[categ_index]]
            x = [x[0] for x in tuple_seg]
            y = [y[1] for y in tuple_seg]
            z = [z[2] for z in tuple_seg]
            ax.scatter(x, y, z, c=color[categ_index], cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    plt.show()


def calc_acc(pred,label):
    '''
    :param pred: shape=[batch_size,n_point] ,numpy
    :param label:
    :return:
    '''
    total_acc = 0
    for i in range(np.shape(pred)[0]):
        total_acc += np.sum([pred[i]==label[i]]) / np.shape(pred)[1]

    return total_acc / np.shape(pred)[0]


def calc_iou(pred,label):
    '''
    计算一个batch里面的iou和acc，所以return的数据为整个batch 的mean data
    若 iou=1，说明label不存在这个class,且pred也没有这个class。
    若 iou=0，说明label存在该class，但pred没有class
    :param pred:
    :param label:
    :return:
    '''
    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                    'bathtub', 'otherfurniture']
    assert 20 == len(CLASS_LABELS), "wrong n_class"

    batch_size = pred.shape[0]  # shape=[batchsize,n_point]
    n_point = pred.shape[1]
    n_class = 20

    cat_table = np.zeros((2,n_class))  # cat_table.shape=[n_class,3]  cat_table[x][0-2]: acc,currency iou,miou


    for i in range(batch_size):
        # per batch
        for cls in range(n_class):
            # per class
            acc_m1 = np.zeros(n_point)
            acc_m2 = np.zeros(n_point)-1
            I = np.sum(np.logical_and(pred[i] == cls, label[i] == cls))
            U = np.sum(np.logical_or(pred[i] == cls, label[i] == cls))
            if U != 0:
                iou = I / U
            else: # label[i] == cls全为0，simple中没有这个class
                iou = 1
            cat_table[1][cls] += iou

            acc_m1[pred[i] == cls] = 1
            acc_m2[label[i] == cls] = 1

            if np.sum(label[i] == cls) == 0:
                # 某些 图的point中不包含某些class，需要把这些class的准确率设为1
                if np.sum([pred[i] == cls]) == 0:
                    cat_table[0][cls] += 1  # 如果pred没有该class，且label没有该class，则acc=1
                else:
                    cat_table[0][cls] += 1  # label没有该class，但pred有，则acc = ?
            else:
                cat_table[0][cls] += np.sum(acc_m1 == acc_m2) / np.sum(label[i] == cls)

    cat_table[1] = cat_table[1] / batch_size
    cat_table[0] = cat_table[0] / batch_size

    return cat_table

def visualizer(points, pred, labels):
    """
    :param points: # shape = (4096,9)  data_batch[2,:,:]
    :param labels:   label_batch[2,:]
    :return:
    """
    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                    'bathtub', 'otherfurniture']

    skip = 1  # Skip every n points

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(121, projection='3d')
    point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
    ax.scatter(points[point_range, 0],  # x
               points[point_range, 1],  # y
               points[point_range, 2],  # z
               c=labels[point_range],  # height data for color
               cmap='Spectral',
               alpha=1,
               s=2,
               marker=".")
    ax.axis('scaled')  # {equal, scaled}

    ax2 = fig.add_subplot(122,projection='3d')
    ax2.scatter(points[point_range, 0],  # x
               points[point_range, 1],  # y
               points[point_range, 2],  # z
               c=pred[point_range],  # height data for color
               cmap='Spectral',
               alpha=1,
               s=2,
               marker=".")
    ax2.axis('scaled')

    # show color bar
    plt.figure(2)
    range_cmap = [[i for i in range(len(CLASS_LABELS))]]
    plt.imshow(range_cmap, cmap='Spectral')
    plt.xticks(range_cmap[0], CLASS_LABELS, rotation=50)
    plt.show()
