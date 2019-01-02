# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:36:23 2018

@author: shen1994
"""

import numpy as np
from math import sqrt, isnan

class HeatMapper:
    
    def __init__(self, 
                 image_shape=(368, 368, 3),
                 stride=8):
        
        self.width = image_shape[0]
        self.height = image_shape[1]
        self.stride = stride

        # this is coordinates of centers of bigger grid
        self.grid_x = np.arange(image_shape[0]//stride)*stride + stride/2.-0.5
        self.grid_y = np.arange(image_shape[1]//stride)*stride + stride/2.-0.5
        self.Y, self.X = np.mgrid[:image_shape[1]:stride, :image_shape[0]:stride]
        
        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
        self.num_parts = len(self.parts)
 
        # this numbers probably copied from matlab they are 1.. based not 0.. based
        limb_from = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
        limb_to = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        limbs_conn = zip(limb_from, limb_to)
        self.limbs_conn = [(fr - 1, to - 1) for (fr, to) in limbs_conn]     
        paf_layers = 2 * len(self.limbs_conn)
        
        self.paf_start = 0
        self.heat_layers = self.num_parts
        num_layers = paf_layers + self.heat_layers + 1
        
        self.heat_start = paf_layers
        self.bkg_start = paf_layers + self.heat_layers
        self.parts_shape = (num_layers, self.height//self.stride, self.width//self.stride)  # 57, 46, 46

        sigma = 7.
        self.double_sigma2 = 2 * sigma * sigma
        self.paf_thre = 8.  # it is original 1.0 * stride in this program
    
    def put_gaussian_maps(self, heatmaps, layer, joints):
    
        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by
        for i in range(joints.shape[0]):
            exp_x = np.exp(-(self.grid_x-joints[i,0])**2/self.double_sigma2)
            exp_y = np.exp(-(self.grid_y-joints[i,1])**2/self.double_sigma2)
    
            exp = np.outer(exp_y, exp_x)
    
            # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[self.heat_start + layer, :, :] = np.maximum(heatmaps[self.heat_start + layer, :, :], exp)
    
    def put_joints(self, heatmaps, joints):
    
        for i in range(self.num_parts):
            visible = joints[:,i,2] < 2
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])
     
    def distances(self, X, Y, x1, y1, x2, y2):
        # classic formula is:
        # d = (x2-x1)*(y1-y)-(x1-x)*(y2-y1)/sqrt((x2-x1)**2 + (y2-y1)**2)
    
        xD = (x2-x1)
        yD = (y2-y1)
        norm2 = sqrt(xD**2 + yD**2)
        dist = (xD*(y1-Y)-(x1-X)*yD).astype(np.float)
        dist /= norm2
    
        return np.abs(dist)
            
    def put_vector_maps(self, heatmaps, layerX, layerY, joint_from, joint_to):

        count = np.zeros(heatmaps.shape[1:], dtype=np.int)

        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2-x1
            dy = y2-y1
            dnorm = sqrt(dx*dx + dy*dy)

            if dnorm==0:  # we get nan here sometimes, it's kills NN
                # TODO: handle it better. probably we should add zero paf, centered paf, or skip this completely
                # print("Parts are too close to each other. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            min_sx = int(round((min_sx - self.paf_thre) / self.stride))
            min_sy = int(round((min_sy - self.paf_thre) / self.stride))
            max_sx = int(round((max_sx + self.paf_thre) / self.stride))
            max_sy = int(round((max_sy + self.paf_thre) / self.stride))

            # check PAF off screen. do not really need to do it with max>grid size
            if max_sy < 0:
                continue

            if max_sx < 0:
                continue

            if min_sx < 0:
                min_sx = 0

            if min_sy < 0:
                min_sy = 0

            #TODO: check it again
            slice_x = slice(min_sx, max_sx) # + 1     this mask is not only speed up but crops paf really. This copied from original code
            slice_y = slice(min_sy, max_sy) # + 1     int g_y = min_y; g_y < max_y; g_y++ -- note strict <

            dist = self.distances(self.X[slice_y,slice_x], self.Y[slice_y,slice_x], x1, y1, x2, y2)
            dist = dist <= self.paf_thre

            # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
            heatmaps[layerX, slice_y, slice_x][dist] = (dist * dx)[dist]  # += dist * dx
            heatmaps[layerY, slice_y, slice_x][dist] = (dist * dy)[dist] # += dist * dy
            count[slice_y, slice_x][dist] += 1

        # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
        # heatmaps[layerX, :, :][count > 0] /= count[count > 0]
        # heatmaps[layerY, :, :][count > 0] /= count[count > 0]
            
    def put_limbs(self, heatmaps, joints):

        for (i,(fr,to)) in enumerate(self.limbs_conn):

            visible_from = joints[:,fr,2] < 2
            visible_to = joints[:,to, 2] < 2
            visible = visible_from & visible_to

            layerX, layerY = (self.paf_start + i*2, self.paf_start + i*2 + 1)
            self.put_vector_maps(heatmaps, layerX, layerY, joints[visible, fr, 0:2], joints[visible, to, 0:2])

    def create_heatmaps(self, joints):
        
        heatmaps = np.zeros(self.parts_shape, dtype=np.float)
        self.put_joints(heatmaps, joints)
        
        sl = slice(self.heat_start, self.heat_start + self.heat_layers)
        heatmaps[self.bkg_start] = 1. - np.amax(heatmaps[sl,:,:], axis=0)
        self.put_limbs(heatmaps, joints)
        
        return heatmaps    
        