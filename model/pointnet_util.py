from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from time import time

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../model'))
sys.path.append("..")
import model.pointconv_util as pointconv_util


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    FPS（质心 通过最远采样点算法 获得）
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids  # 相邻的球，球质心


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Simpling ：selects a set of points from input points (确定局部区域的图心）,
            迭代最远点采样（FPS）来选择点x1，x2...的子集 ，（距离其余的子集在欧几里得空间上距离最远)

    Groupling：分组层通过查找质心周围的“邻近”点来构建局部区域集。
            input:大小为N*（d + C）的点集和大小为 N_Id的一组质心的坐标
            output:groups of point sets of size N0 × K × (d + C),where each group corresponds to a local region and K is the number of points in the neighborhood of centroid points

        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, K, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction_PointConv(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction_PointConv, self).__init__()
        self.npoint = npoint  # 这个npoint是指这个layer的质心坐标的数量，不是总points数
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()  # 用于存储layer顺序的list
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:  # 生成mlp层[x1,x2,x3]
            self.mlp_convs.append(
                nn.Conv2d(last_channel, out_channel, 1))  # 用Conv2d来做FC，Conv2d(in_channels,out_channel,k_size)
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.mlp = mlp

        self.c_mid = 32
        self.weight_net_hidden = nn.Conv2d(in_channels=3,out_channels=self.c_mid,kernel_size=(1,1))
        self.weight_net_hidden_bn = nn.BatchNorm2d(self.c_mid)

        # mlp_inverse density scale = [16,1]
        self.nonlinear_transform_1 = nn.Conv2d(1,16,(1,1))
        self.nonlinear_transform_1_bn1 = nn.BatchNorm2d(16)
        self.nonlinear_transform_2 = nn.Conv2d(16,1,(1,1))
        self.nonlinear_transform_2_bn2 = nn.BatchNorm2d(1)

        # out put conv
        self.out_conv = nn.Conv2d(in_channel,out_channels=mlp[-1],kernel_size=(1,self.c_mid))
        self.out_bn = nn.BatchNorm2d(mlp[-1])

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]  # batch_size, 3, n_points
            points: input points data, [B, D, N]  # batch_size, 6, n_points
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        grouped_xyz = 0
        if self.group_all:  # PointNet2SemSeg 里面全是False
            new_xyz, new_points = sample_and_group_all(xyz, points)
            #         new_xyz: sampled points position data, [B, 1, C]
            #         new_points: sampled points data, [B, 1, N, C+D]
        else:
            new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points,returnfps=True)
        #         new_xyz: sampled points position data, [B, npoint, C]
        #         new_points: sampled points data, [B, npoint, n_simple, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # # 下面这个是每个set abstraction block的pointnet layer

        # [8,9,32,1024]  9=in_channel
        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     new_points = F.relu(bn(conv(new_points)))  # conv2d：Input:`(N, C_{in}, H_{in}, W_{in})`
        #
        # # [8,64,32,1024]  64=mlp[-1]
        # new_points = torch.max(new_points, 2)[0]  # 这个是max pooling layer ([0]是softmax的值，[1]是index）
        # new_xyz = new_xyz.permute(0, 2, 1)

        # new_xyz = [batch_size, npoint,3]
        # new_points = [batch_size, channel(in_channel), nsample, npoint]
        # grouped_xyz = [8, 1024, 32, 3]

        grouped_feature = new_points
        grouped_feature[grouped_feature < 1e-10] = 1e-10

        # density [batch,npoint,nsample,1]
        density = pointconv_util.kernel_density_estimation_ball(grouped_xyz)# KDE! INPUT:xyz[B,3,npoint](算每个
        density[density < 1e-10] = 1e-10
        inverse_density = torch.div(1.,density)  # div
        # grouped_density = torch.gather(inverse_density, 0,idx) # (batch_size, npoint, nsample, 1)  tf.gather_nd

        inverse_max_density = torch.max(inverse_density, dim = 2, keepdim = True)[0]  # tf.reduce_max
        density_scale = torch.div(inverse_density, inverse_max_density)  # density_scale = [8, 1024, 32, 1]


        # weight = pointconv_util.weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        grouped_xyz = torch.Tensor.permute(grouped_xyz,[0,3,1,2])  # [B,npoint,nsample,3] => [B,3,npoint,nsample]
        weight = F.relu(self.weight_net_hidden_bn(self.weight_net_hidden(grouped_xyz)))  #weight [8, 32, 1024, 32]

        density_scale = torch.Tensor.permute(density_scale,[0,3,1,2])
        density_scale1 = F.relu(self.nonlinear_transform_1_bn1(self.nonlinear_transform_1(density_scale)))
        density_scale = torch.sigmoid(self.nonlinear_transform_2_bn2(self.nonlinear_transform_2(density_scale1)))

        # density_scale = [8, 1, 1024, 32]
        # grouped_feature = [8, 9, 32, 1024]
        grouped_feature = torch.Tensor.permute(grouped_feature,[0,1,3,2])
        new_points = torch.mul(grouped_feature, density_scale)  # 用广播机制代替tile

        # new_points = [8, 9, 1024, nsample]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3]) # [8,1024,9,32]
        # weight [8, 32, 1024, nsample]
        weight = torch.Tensor.permute(weight,[0,2,3,1])

        new_points = torch.matmul(new_points, weight)  # 结果为  C_in * C_mid

        # new_points = [8, 1024, 9, c_mid]
        new_points = torch.Tensor.permute(new_points,[0,2,1,3])

        new_points = self.out_bn(self.out_conv(new_points))

        # new_points = [8, 64, 1024, 1]
        new_points = torch.squeeze(new_points,dim=-1)

        new_xyz = torch.Tensor.permute(new_xyz,[0,2,1])

        # new_xyz = [batch_size, channel(3), npoint]
        # new_points= [batch_size, channel(mlp[-1]), npoint(质心数)
        return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint  # 这个npoint是指这个layer的质心坐标的数量，不是总points数
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()  # 用于存储layer顺序的list
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:  # 生成mlp层[x1,x2,x3]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))  # 用Conv2d来做FC，Conv2d(in_channels,out_channel,k_size)
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input00:
            xyz: input points position data, [B, C, N]  # batch_size, 3, n_points
            points: input points data, [B, D, N]  # batch_size, 6, n_points
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:  # PointNet2SemSeg 里面全是False
            new_xyz, new_points = sample_and_group_all(xyz, points)
            #         new_xyz: sampled points position data, [B, 1, C]
            #         new_points: sampled points data, [B, 1, N, C+D]
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        #         new_xyz: sampled points position data, [B, npoint, C]
        #         new_points: sampled points data, [B, npoint, n_simple, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]


        # 下面这个是每个set abstraction block的pointnet layer
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))   # conv2d：Input:`(N, C_{in}, H_{in}, W_{in})`

        new_points = torch.max(new_points, 2)[0] # 这个是max pooling layer ([0]是softmax的值，[1]是index）
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points

# SemSeg没用到，这是multiply Scale Group的方法
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

# fp,upsample
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points

