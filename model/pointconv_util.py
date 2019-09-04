from __future__ import absolute_import

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class pointconv_old(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(pointconv_old,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.c_mid = 32
        self.weight_net_hidden = nn.Conv2d(in_channels=3,out_channels=self.c_mid,kernel_size=(1,1))
        self.weight_net_hidden_bn = nn.BatchNorm2d(self.c_mid)

        # mlp_inverse density scale = [16,1]
        self.nonlinear_transform_1 = nn.Conv2d(1,16,(1,1))
        self.nonlinear_transform_1_bn1 = nn.BatchNorm2d(16)
        self.nonlinear_transform_2 = nn.Conv2d(16,1,(1,1))
        self.nonlinear_transform_2_bn2 = nn.BatchNorm2d(1)

        # out put conv
        self.out_conv = nn.Conv2d(in_channels,out_channels=out_channels,kernel_size=(1,self.c_mid))
        self.out_bn = nn.BatchNorm2d(out_channels)


    def forward(self, new_xyz, grouped_xyz, grouped_feature):
        '''
        :param new_xyz: [B, 3, npoint]
        :param grouped_xyz: [B, npoint, nsample, 3]
        :param grouped_feature: [B, in_channel, nsample, npoint]
        :return: new_xyz: [B, 3, npoint]
                new_points: [B, out)
        '''
        # density [batch,npoint,nsample,1]
        density = kernel_density_estimation_ball(grouped_xyz)  # KDE! INPUT:xyz[B,3,npoint](算每个
        density[density < 1e-10] = 1e-10
        inverse_density = torch.div(1., density)  # div
        # grouped_density = torch.gather(inverse_density, 0,idx) # (batch_size, npoint, nsample, 1)  tf.gather_nd

        inverse_max_density = torch.max(inverse_density, dim=2, keepdim=True)[0]  # tf.reduce_max
        density_scale = torch.div(inverse_density, inverse_max_density)  # density_scale = [8, 1024, 32, 1]

        # weight = pointconv_util.weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        grouped_xyz = torch.Tensor.permute(grouped_xyz, [0, 3, 1, 2])  # [B,npoint,nsample,3] => [B,3,npoint,nsample]
        weight = F.relu(self.weight_net_hidden_bn(self.weight_net_hidden(grouped_xyz)))  # weight [8, 32, 1024, 32]

        density_scale = torch.Tensor.permute(density_scale, [0, 3, 1, 2])
        density_scale1 = F.relu(self.nonlinear_transform_1_bn1(self.nonlinear_transform_1(density_scale)))
        density_scale = torch.sigmoid(self.nonlinear_transform_2_bn2(self.nonlinear_transform_2(density_scale1)))

        # density_scale = [8, 1, 1024, 32]
        # grouped_feature = [8, 9, 32, 1024]
        grouped_feature = torch.Tensor.permute(grouped_feature, [0, 1, 3, 2])
        new_points = torch.mul(grouped_feature, density_scale)  # 用广播机制代替tile

        # new_points = [8, 9, 1024, nsample]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3])  # [8,1024,9,32]
        # weight [8, 32, 1024, nsample]
        weight = torch.Tensor.permute(weight, [0, 2, 3, 1])

        new_points = torch.matmul(new_points, weight)  # 结果为  C_in * C_mid

        # new_points = [8, 1024, 9, c_mid]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3])

        new_points = self.out_bn(self.out_conv(new_points))

        # new_points = [8, 64, 1024, 1]
        new_points = torch.squeeze(new_points, dim=-1)

        new_xyz = torch.Tensor.permute(new_xyz, [0, 2, 1])

        return new_xyz,new_points


class depointconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(depointconv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.c_mid = 32
        self.weight_net_hidden = nn.Conv2d(in_channels=3,out_channels=self.c_mid,kernel_size=(1,1))
        self.weight_net_hidden_bn = nn.BatchNorm2d(self.c_mid)

        # mlp_inverse density scale = [16,1]
        self.nonlinear_transform_1 = nn.Conv2d(1,16,(1,1))
        self.nonlinear_transform_1_bn1 = nn.BatchNorm2d(16)
        self.nonlinear_transform_2 = nn.Conv2d(16,1,(1,1))
        self.nonlinear_transform_2_bn2 = nn.BatchNorm2d(1)

        # out put conv
        self.out_conv = nn.Conv2d(in_channels,out_channels=out_channels,kernel_size=(1,self.c_mid))
        self.out_bn = nn.BatchNorm2d(out_channels)


    def forward(self, xyz,points,npoint,radius,nsample):
        """
        Input:
            xyz: input points position data, [B, C, N]  # batch_size, 3, n_points
            points: input points data, [B, D, N]  # batch_size, 6, n_points
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0,2,1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(npoint, radius, nsample, xyz,points, returnfps=True)
        grouped_feature = new_points.permute(0, 3, 2, 1)  # [B, D, nsample,npoint]

        # grouped_feature = grouped_feature[:,3:,:,:]  # 去掉前3个channel（代表了xyz特征）
        # density [batch,npoint,nsample,1]
        density = kernel_density_estimation_ball(grouped_xyz)  # KDE! INPUT:xyz[B,3,npoint](算每个
        density[density < 1e-10] = 1e-10
        inverse_density = torch.div(1., density)  # div
        # grouped_density = torch.gather(inverse_density, 0,idx) # (batch_size, npoint, nsample, 1)  tf.gather_nd

        inverse_max_density = torch.max(inverse_density, dim=2, keepdim=True)[0]  # tf.reduce_max
        density_scale = torch.div(inverse_density, inverse_max_density)  # density_scale = [8, 1024, 32, 1]

        # weight = pointconv_util.weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        grouped_xyz = torch.Tensor.permute(grouped_xyz, [0, 3, 1, 2])  # [B,npoint,nsample,3] => [B,3,npoint,nsample]
        weight = F.relu(self.weight_net_hidden_bn(self.weight_net_hidden(grouped_xyz)))  # weight [8, 32, 1024, 32]

        density_scale = torch.Tensor.permute(density_scale, [0, 3, 1, 2])
        density_scale1 = F.relu(self.nonlinear_transform_1_bn1(self.nonlinear_transform_1(density_scale)))
        density_scale = torch.sigmoid(self.nonlinear_transform_2_bn2(self.nonlinear_transform_2(density_scale1)))

        # density_scale = [8, 1, 1024, 32]
        # grouped_feature = [8, 9, 32, 1024]
        grouped_feature = torch.Tensor.permute(grouped_feature, [0, 1, 3, 2])
        new_points = torch.mul(grouped_feature, density_scale)  # 用广播机制代替tile

        # new_points = [8, 9, 1024, nsample]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3])  # [8,1024,9,32]
        # weight [8, 32, 1024, nsample]
        weight = torch.Tensor.permute(weight, [0, 2, 3, 1])

        new_points = torch.matmul(new_points, weight)  # 结果为  C_in * C_mid
        # new_points = [8, 1024, 9, c_mid]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3])

        new_points = self.out_bn(self.out_conv(new_points))

        # new_points = [8, 64, 1024, 1]
        new_points = torch.squeeze(new_points, dim=-1)

        new_xyz = torch.Tensor.permute(new_xyz, [0, 2, 1])

        return new_points


class pointconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(pointconv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.c_mid = 32
        self.weight_net_hidden = nn.Conv2d(in_channels=3,out_channels=self.c_mid,kernel_size=(1,1))
        self.weight_net_hidden_bn = nn.BatchNorm2d(self.c_mid)

        # mlp_inverse density scale = [16,1]
        self.nonlinear_transform_1 = nn.Conv2d(1,16,(1,1))
        self.nonlinear_transform_1_bn1 = nn.BatchNorm2d(16)
        self.nonlinear_transform_2 = nn.Conv2d(16,1,(1,1))
        self.nonlinear_transform_2_bn2 = nn.BatchNorm2d(1)

        # out put conv
        self.out_conv = nn.Conv2d(in_channels,out_channels=out_channels,kernel_size=(1,self.c_mid))
        self.out_bn = nn.BatchNorm2d(out_channels)


    def forward(self, xyz,points,npoint,radius,nsample):
        """
        Input:
            xyz: input points position data, [B, C, N]  # batch_size, 3, n_points
            points: input points data, [B, D, N]  # batch_size, 6, n_points
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0,2,1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz, new_points, grouped_xyz, fps_idx = sample_and_group(npoint, radius, nsample, xyz,
                                                                     points, returnfps=True)
        grouped_feature = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]


        # density [batch,npoint,nsample,1]
        density = kernel_density_estimation_ball(grouped_xyz)  # KDE! INPUT:xyz[B,3,npoint](算每个
        density[density < 1e-10] = 1e-10
        inverse_density = torch.div(1., density)  # div
        # grouped_density = torch.gather(inverse_density, 0,idx) # (batch_size, npoint, nsample, 1)  tf.gather_nd

        inverse_max_density = torch.max(inverse_density, dim=2, keepdim=True)[0]  # tf.reduce_max
        density_scale = torch.div(inverse_density, inverse_max_density)  # density_scale = [8, 1024, 32, 1]

        # weight = pointconv_util.weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        grouped_xyz = torch.Tensor.permute(grouped_xyz, [0, 3, 1, 2])  # [B,npoint,nsample,3] => [B,3,npoint,nsample]
        weight = F.relu(self.weight_net_hidden_bn(self.weight_net_hidden(grouped_xyz)))  # weight [8, 32, 1024, 32]

        density_scale = torch.Tensor.permute(density_scale, [0, 3, 1, 2])
        density_scale1 = F.relu(self.nonlinear_transform_1_bn1(self.nonlinear_transform_1(density_scale)))
        density_scale = torch.sigmoid(self.nonlinear_transform_2_bn2(self.nonlinear_transform_2(density_scale1)))

        # density_scale = [8, 1, 1024, 32]
        # grouped_feature = [8, 9, 32, 1024]
        grouped_feature = torch.Tensor.permute(grouped_feature, [0, 1, 3, 2])
        new_points = torch.mul(grouped_feature, density_scale)  # 用广播机制代替tile

        # new_points = [8, 9, 1024, nsample]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3])  # [8,1024,9,32]
        # weight [8, 32, 1024, nsample]
        weight = torch.Tensor.permute(weight, [0, 2, 3, 1])

        new_points = torch.matmul(new_points, weight)  # 结果为  C_in * C_mid

        # new_points = [8, 1024, 9, c_mid]
        new_points = torch.Tensor.permute(new_points, [0, 2, 1, 3])

        new_points = self.out_bn(self.out_conv(new_points))

        # new_points = [8, 64, 1024, 1]
        new_points = torch.squeeze(new_points, dim=-1)

        new_xyz = torch.Tensor.permute(new_xyz, [0, 2, 1])

        return new_xyz,new_points


def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        new_xyz: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''

    # def index_points(points, idx):
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


def kernel_density_estimation_ball(xyz):
    '''
    简单计算每个local point中K个point到P0的距离(其中P0为每个nsimple里面第一位）
    :param xyz: [batch,npoint,nsample,3]
    :return: density [batch,npoint,nsample,1]
    '''
    # xyz = torch.var(xyz,dim=2)
    for i in range(xyz.shape[2]):
        xyz[:,:,i,:] = xyz[:,:,i,:] - xyz[:,:,0,:]
    density = torch.sum(xyz,dim=-1,keepdim=True)

    return density


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
        #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        # 用于去掉（上面这行）concatenate xyz这3channel，使feature 保持RGB+local（原本是xyz+RGB+local）【这个local可以在预处理时给加上】
        new_points = grouped_points
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


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


# def query_ball_point_temp(radius, nsample, xyz, new_xyz):
#     """
#     from pointnet
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, C]
#         new_xyz: query points, [B, S, C]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     '''
#     // input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
#     // output: idx (b,m,nsample)
#     '''
#     print(xyz.shape)
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
#     sqrdists = square_distance_temp(new_xyz, xyz)
#     group_idx[sqrdists > radius ** 2] = N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#     return group_idx
#
#
# def square_distance_temp(src, dst):
#     """
#     from pointnet
#     Calculate Euclid distance between each two points.
#
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist


# def kernel_density_estimation_ball2(xyz, radius, sigma, N_points = 128, is_norm = False):
#     '''
#
#     :param xyz: xyz
#     :param radius:
#     :param sigma:
#     :param N_points:
#     :param is_norm:
#     :return:
#     '''
#     print(xyz.shape)
#     # 已经算了 ball point了
#     # idx, xyz_cnt = query_ball_point_temp(radius, N_points, xyz, xyz)
#
#     g_xyz = group_point(xyz, idx)
#     g_xyz -= torch.Tensor.repeat(tf.expand_dims(xyz, 2), [1, 1, N_points, 1])
#
#     R = torch.sqrt(sigma)
#     xRinv = torch.div(g_xyz, R)
#     quadform = torch.sum(torch.sqrt(xRinv), dim = -1)
#     logsqrtdetSigma = torch.log(R) * 3
#     mvnpdf = torch.exp(-0.5 * quadform - logsqrtdetSigma - 3 * torch.log(2 * 3.1415926) / 2)
#
#     first_val, _ = torch.split(mvnpdf, [1, N_points - 1], dim = 2)
#
#     mvnpdf = torch.sum(mvnpdf, dim = 2, keepdim = True)
#
#     num_val_to_sub = tf.expand_dims(tf.cast(tf.subtract(N_points, xyz_cnt), dtype = tf.float32), axis = -1)
#
#     val_to_sub = torch.mul(first_val, num_val_to_sub)
#
#     mvnpdf = tf.subtract(mvnpdf, val_to_sub)
#
#     scale = torch.div(1.0, tf.expand_dims(tf.cast(xyz_cnt, dtype = tf.float32), axis = -1))
#     density = torch.mul(mvnpdf, scale)
#
#     if is_norm:
#         #grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
#         density_max = tf.reduce_max(density, axis = 1, keepdims = True)
#         density = tf.div(density, density_max)
#
#     return density


# def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn="tf.nn.relu"):
#
#     net = xyz
#     for i, num_hidden_units in enumerate(hidden_units):
#         net = torch.nn.Conv2d(net, num_hidden_units, [1, 1],
#                             padding = 'VALID', stride=[1, 1],
#                             bn = True, is_training = is_training, activation_fn=activation_fn,
#                             scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
#         net = torch.nn.Conv2d()
#     return net



# def nonlinear_transform(data_in, mlp, scope, is_training, bn_decay=None, weight_decay = None, activation_fn = "tf.nn.relu"):
#
#     with tf.variable_scope(scope) as sc:
#
#         net = data_in
#         l = len(mlp)
#         if l > 1:
#             for i, out_ch in enumerate(mlp[0:(l-1)]):
#                 net = tf_util.conv2d(net, out_ch, [1, 1],
#                                     padding = 'VALID', stride=[1, 1],
#                                     bn = True, is_training = is_training, activation_fn=tf.nn.relu,
#                                     scope = 'nonlinear%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
#
#                 #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear%d'%(i))
#         net = tf_util.conv2d(net, mlp[-1], [1, 1],
#                             padding = 'VALID', stride=[1, 1],
#                             bn = False, is_training = is_training,
#                             scope = 'nonlinear%d'%(l-1), bn_decay=bn_decay,
#                             activation_fn=tf.nn.sigmoid, weight_decay = weight_decay)
#
#     return net

