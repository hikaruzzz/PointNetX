import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from model.pointnet_util import PointNetSetAbstraction_PointConv,PointNetFeaturePropagation_PointConv

class PointNet2ClsMsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x,l3_points


class PointNet2ClsSsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsSsg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x


class PointNet2PartSeg(nn.Module): #TODO part segmentation tasks
    def __init__(self, num_classes):
        super(PointNet2PartSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, feat

class PointNet2PartSeg_msg_one_hot(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2PartSeg_msg_one_hot, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 0+3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, norm_plt, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.size()
        l0_xyz = xyz
        l0_points = norm_plt
        l1_xyz, l1_points = self.sa1(l0_xyz, norm_plt)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


# main model
class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        # npoint, radius, nsample, in_channel, mlp, group_all
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)  # in_channels, out_channels, kernel_size
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz,points):
        # set abstraction
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # segmentation部分的 反卷积
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # skip link concatenation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points) # 论文：最后一个layer是没有skip link的
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # 固定搭配？
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)  # 这里不要log_softmax，在调用model时加上ok？
        x = x.permute(0, 2, 1)
        return x

# main model
class PointNet2SemSeg_PointConv(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg_PointConv, self).__init__()
        # npoint, radius, nsample, in_channel, mlp, group_all
        # 这里的 inchannel是指point feature的，已经去掉sample&group函数中的concatenate xyz了。
        self.sa1 = PointNetSetAbstraction_PointConv(1024, 0.1, 32, 6, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction_PointConv(256, 0.2, 32, 64, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction_PointConv(64, 0.4, 32, 128, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction_PointConv(16, 0.8, 32, 256, [256, 256, 512], False)

        # 这个6+3，是指point feature的channel是6+3，明明是RGB+local的，还concatenate xyz？？？
        # self.sa1 = PointNetSetAbstraction_PointConv(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        # self.sa2 = PointNetSetAbstraction_PointConv(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction_PointConv(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction_PointConv(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation_PointConv(768, [256, 256], 64, 0.4, 32)  # 768 = s4的512output + s3的256output channel
        self.fp3 = PointNetFeaturePropagation_PointConv(384, [256, 256], 256, 0.2, 32)   # 后面三个是跟随对应skip sa中最大npoint那个的采样值
        self.fp2 = PointNetFeaturePropagation_PointConv(320, [256, 128], 1024, 0.1, 32)
        self.fp1 = PointNetFeaturePropagation_PointConv(128, [128, 128, 128], 8192, 0.05, 32)
        self.conv1 = nn.Conv1d(128, 128, 1)  # in_channels, out_channels, kernel_size
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz,points):
        # set abstraction
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # segmentation部分的 反卷积
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # skip link concatenation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points) # 论文：最后一个layer是没有skip link的

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # 固定搭配？
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)  # 这里不要log_softmax，在调用model时加上ok？
        x = x.permute(0, 2, 1)

        return x


if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input = torch.randn((8,3,2048))
    # label = torch.randn(8,16)
    #model = PointNet2PartSeg_msg_one_hot(num_classes=16)
    # output = model(input, input, label)
    points = torch.randn((8, 9, 2333))
    label = torch.randn(8, 1024)
    model = PointNet2SemSeg_PointConv(num_classes=16)
    # model = PointNet2SemSeg(num_classes=16)
    output = model(points[:,:3,:],points[:,3:,:])
    print("model output",output.size())

