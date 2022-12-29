#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def clip_point_cloud(point_cloud, radius, order=np.inf, normalize=False):
    # point_cloud: xyz x N
    norms = np.linalg.norm(point_cloud[:3], axis=0, ord=order)
    maxnorm = np.max(norms) if normalize else 1.0
    inds = (norms / maxnorm) <= radius
    return point_cloud[:, inds]

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def transRot(V, rot, t, asform="quaternion", reverse=False):
    # translates and rotates a given point cloud by specified parameters
    # V is a point cloud,
    # rot is a rotation, which can be given in 3 forms determined by asform
    #   1) a quaternion (versor) rotation,
    #   2) a rotation matrix
    #   3) a set of Euler angles
    # t is a 3D translation
 
    # first map all rotations to matrices using scipy's nice little rotation module!
    if asform == "quaternion":
        # rot is a quaternion
        r = np.asarray(Rotation.as_matrix(Rotation.from_quat(rot)))
    elif asform == "matrix":
        # rot is a rotation matrix
        r = rot
    elif asform == "angles":
        # rot is 3 angles -- x, y, z
        r = np.asarray(Rotation.as_matrix(Rotation.from_euler('xyz',rot,degrees=True)))
 
    N = V.shape[1]
    # it is just a matrix multiply and addition

    if reverse:
        V = r.T @ (V - np.tile(np.array([t]).T,(1,N)))
    else:
        V = np.tile(np.array([t]).T,(1,N))+ r@V
    return V

