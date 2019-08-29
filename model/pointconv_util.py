from __future__ import absolute_import

import numpy as np
import os
import sys
import torch


def query_ball_point_temp(radius, nsample, xyz, new_xyz):
    """
    from pointnet
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    '''
    // input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
    // output: idx (b,m,nsample)
    '''
    print(xyz.shape)
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance_temp(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance_temp(src, dst):
    """
    from pointnet
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


def kernel_density_estimation_ball2(xyz, radius, sigma, N_points = 128, is_norm = False):
    '''
    
    :param xyz: xyz
    :param radius: 
    :param sigma: 
    :param N_points: 
    :param is_norm: 
    :return: 
    '''
    print(xyz.shape)
    # 已经算了 ball point了
    # idx, xyz_cnt = query_ball_point_temp(radius, N_points, xyz, xyz)

    g_xyz = group_point(xyz, idx)
    g_xyz -= torch.Tensor.repeat(tf.expand_dims(xyz, 2), [1, 1, N_points, 1])

    R = torch.sqrt(sigma)
    xRinv = torch.div(g_xyz, R)
    quadform = torch.sum(torch.sqrt(xRinv), dim = -1)
    logsqrtdetSigma = torch.log(R) * 3
    mvnpdf = torch.exp(-0.5 * quadform - logsqrtdetSigma - 3 * torch.log(2 * 3.1415926) / 2)

    first_val, _ = torch.split(mvnpdf, [1, N_points - 1], dim = 2)

    mvnpdf = torch.sum(mvnpdf, dim = 2, keepdim = True)

    num_val_to_sub = tf.expand_dims(tf.cast(tf.subtract(N_points, xyz_cnt), dtype = tf.float32), axis = -1)

    val_to_sub = torch.mul(first_val, num_val_to_sub)

    mvnpdf = tf.subtract(mvnpdf, val_to_sub)

    scale = torch.div(1.0, tf.expand_dims(tf.cast(xyz_cnt, dtype = tf.float32), axis = -1))
    density = torch.mul(mvnpdf, scale)

    if is_norm:
        #grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
        density_max = tf.reduce_max(density, axis = 1, keepdims = True)
        density = tf.div(density, density_max)

    return density


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

