import os
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
import time

from data_utils.ScanNetLoader import Scannetloader
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import calc_acc,calc_iou,visualizer
from model.pointnet2 import PointNet2SemSeg,PointNet2SemSeg_PointConv
from config import *

# command: CUDA_VISIBLE_DEVICES=0,1 python train_sem.py

# ScanNet
CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
seg_classes = {cls: i for i,cls in enumerate(CLASS_LABELS)}
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


# train
torch.manual_seed(1280)
torch.cuda.manual_seed(1280)
np.random.seed(1280)


def train():
    # read device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device),end="\n")
    # load dataset
    data_loader_train = Scannetloader(scannet_file_root,max_point,n_class,is_train=True)
    data_loader_val   = Scannetloader(scannet_file_root,max_point,n_class,is_train=False)

    train_loader = torch.utils.data.DataLoader(data_loader_train,batch_size=batch_size,shuffle=True,num_workers=n_workers)
    val_loader   = torch.utils.data.DataLoader(data_loader_val,batch_size=batch_size,shuffle=True,num_workers=n_workers)

    # set model running devices
    #model = PointNet2SemSeg(n_class)
    model = PointNet2SemSeg_PointConv(n_class)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    #?torch.backends.cudnn.benchmark = True
    print("usable gpu nums: {}".format(torch.cuda.device_count()))


    # set optimizer, lr_scheduler, loss function
    optimizer = None
    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay_rate)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-08)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    # criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = F.nll_loss()

    # load checkpoints
    last_best_iou = -100.
    load_ckpt_path = os.path.join("checkpoints",load_ckpt_name)
    start_epoch = 0
    if is_load_checkpoints:
        if torch.cuda.is_available():
            checkpoint = torch.load(load_ckpt_path)
        else:
            checkpoint = torch.load(load_ckpt_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        last_best_iou = checkpoint['best_iou']
        start_epoch = checkpoint['epoch']

        print('Checkpoint resume success... last iou:{:.4%},last epoch:{}'.format(last_best_iou,start_epoch))

    # train epoch
    best_iou = last_best_iou
    time_epoch = time.time()
    i = 0
    print("training...")
    for epoch in range(start_epoch,max_epoch):
        time_step = time.time()
        lr_scheduler.step(epoch=epoch)
        for step, batch in enumerate(train_loader):
            model.train()
            points = batch[0].to(device)
            targets = batch[1].long().to(device)

            points = points.transpose(2,1)
            optimizer.zero_grad()
            # print(points.shape) = [3, 9, 8192]
            pred = model(points[:, :3, :], points[:, 3:, :])
            pred = pred.contiguous().view(-1, n_class)
            # targets = targets.view(-1, 1)[:, 0]
            targets = targets.contiguous().view(-1)

            #loss = F.cross_entropy(input=pred,target=targets) # log_softmax + nll_loss
            loss = F.nll_loss(pred, targets)  # softmax + Cross-Entropy cost

            cur_loss = np.float(loss.cpu().detach().numpy()) # 不要用.data,会导致tensor flag: requires_grad=False,终止求导

            loss.backward()
            optimizer.step()

            writer.add_scalars('Train_record/Loss per X step', {'loss': cur_loss}, step+epoch*len(train_loader))
            cur_time = time.time()
            print("\rtrain | epoch: {}/{} | step:{}/{} | {} s/step | time used:{:d}s | currency loss:{:.4}".
                  format(epoch,max_epoch,step+1,len(train_loader),int((cur_time-time_step)/(step+1)),int(cur_time-time_epoch),cur_loss),end='',flush=True)
        # val
        print("\nvaluing")
        with torch.no_grad():
            model.eval()
            cat_table = 0
            total_cat_table = np.zeros((2,n_class))  # [0]:acc,[1]:miou
            time_step_val = time.time()
            for step_val, batch_val in enumerate(val_loader):
                #print("read batch time used:",time.time()-time_step)
                points_val = batch_val[0].to(device)
                labels_val = batch_val[1].cpu().data.numpy()
                points_val = points_val.transpose(2,1)

                pred_val = model(points_val[:, :3, :], points_val[:, 3:, :])
                # pred_val.shape = [batch_size,n_point,n_class]
                pred_val = F.softmax(pred_val,dim=2).cpu().data.numpy()
                pred_val = np.argmax(pred_val,axis=2)

                #visualizer(points_val.transpose(2,1)[0],pred_val[0],labels_val[0])

                cat_table = calc_iou(pred_val,labels_val)
                total_cat_table += cat_table

                cur_time = time.time()
                print("\rvalue | step:{}/{} | {} s/step | mean iou:{:.4} | mean acc:{:.4}".
                      format(step_val + 1, len(val_loader), int((cur_time - time_step_val) / (step_val + 1)),
                              np.mean(cat_table[1]),np.mean(cat_table[0])), end='', flush=True)

            total_cat_table = total_cat_table / len(val_loader)
            # total_cat_table = total_cat_table / (step_val+1)
            mean_iou = np.mean(total_cat_table[1])
            mean_acc = np.mean(total_cat_table[0])
            print("\ntrue accuracy:{:.3f}".format(calc_acc(pred_val,labels_val)))
            print("\naccuracy: {:.3f} | mean iou: {:.3f}".format(mean_acc,mean_iou))
            writer.add_scalars('Val_record/mIoU', {'accuracy': mean_acc}, epoch)
            writer.add_scalars('val_record/mIoU', {'mIoU': mean_iou}, epoch)
            print("class name \tmean accuracy\tmIoU")
            for i in range(n_class):
                print("%-18s %.3f \t%.3f"%(CLASS_LABELS[i],total_cat_table[0][i],total_cat_table[1][i]))
            if mean_iou > best_iou:
                best_iou = mean_iou
                state ={
                    "epoch":epoch + 1,
                    "model_state":model.state_dict(),
                    "best_iou":best_iou,
                }
                if not os.path.isdir('./checkpoints'):
                    os.mkdir('./checkpoints')
                save_path = os.path.join('./checkpoints', "eph_{}_iou_{:.2%}.ckpt.pth".format(epoch+1,best_iou))
                torch.save(state,save_path)
                print("checkpoint saved success")

    writer.close()


if __name__ == "__main__":
    writer = SummaryWriter()
    torch.backends.cudnn.benchmark = True
    train()
