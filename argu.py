# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:17:46 2018

@author: shen1994
"""

import cv2
import numpy as np
from math import cos, sin, pi

class Argument:
    
    def __init__(self, image_shape=(368, 368, 3), stride=8,
                 flip=False, degree=0., crop=(0, 0), scale=1.):
        
        self.width = image_shape[0]
        self.height = image_shape[1]

        self.mask_shape = (self.height//stride, self.width//stride)  # 46, 46
        
        parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
        parts_dict = dict(zip(parts, range(len(parts))))
        self.leftParts  = [ parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"] ]
        self.rightParts = [ parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"] ]

        self.flip = flip
        self.degree = degree
        self.crop = crop
        self.scale = scale
        
        self.target_dist = 0.6
        self.scale_prob = 1   # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
        self.scale_min = 0.5
        self.scale_max = 1.1
        self.max_rotate_degree = 40.
        self.center_perterb_max = 40.
        self.flip_prob = 0.5
        
    def random(self):
        self.flip = np.random.uniform(0., 1.) > self.flip_prob
        self.degree = np.random.uniform(-1., 1.) * self.max_rotate_degree
        self.scale = (self.scale_max - self.scale_min)*np.random.uniform(0.,1.)+self.scale_min \
            if np.random.uniform(0.,1.) > self.scale_prob else 1. # TODO: see 'scale improbability' TODO above
        x_offset = int(np.random.uniform(-1., 1.) * self.center_perterb_max)
        y_offset = int(np.random.uniform(-1., 1.) * self.center_perterb_max)
        self.crop = (x_offset, y_offset)

    def unrandom(self):
        self.flip = False
        self.degree = 0.
        self.scale = 1.
        self.crop = (0, 0)
        
    def affline(self, center, scale_self):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards

        A = self.scale * cos(self.degree / 180. * pi )
        B = self.scale * sin(self.degree / 180. * pi )

        scale_size = self.target_dist / scale_self * self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array( [[ 1., 0., -center_x],
                                 [ 0., 1., -center_y ],
                                 [ 0., 0., 1. ]] )

        rotate = np.array( [[ A, B, 0 ],
                           [ -B, A, 0 ],
                           [  0, 0, 1. ] ])

        scale = np.array( [[ scale_size, 0, 0 ],
                           [ 0, scale_size, 0 ],
                           [  0, 0, 1. ] ])

        flip = np.array( [[ -1 if self.flip else 1., 0., 0. ],
                          [ 0., 1., 0. ],
                          [ 0., 0., 1. ]] )

        center2center = np.array( [[ 1., 0., self.width//2],
                                   [ 0., 1., self.height//2 ],
                                   [ 0., 0., 1. ]] )

        # order of combination is reversed
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)

        return combined[0:2]

    def transform(self, image, mask, meta, is_random=True):
        
        if is_random:
            self.random()
        else:
            self.unrandom()
            
        M = self.affline(meta['objpos'][0], meta['scale_provided'][0])
        
        # TODO: need to understand this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
        image = cv2.warpAffine(image, M, (self.height, self.width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        mask = cv2.warpAffine(mask, M, (self.height, self.width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        mask = cv2.resize(mask, self.mask_shape, interpolation=cv2.INTER_CUBIC)  # TODO: should be combined with warp for speed
        mask = mask.astype(np.float) / 255.

        # warp key points
        #TODO: joint could be cropped by augmentation, in this case we should mark it as invisible.
        #update: may be we don't need it actually, original code removed part sliced more than half totally, may be we should keep it
        original_points = meta['keypoints'].copy()
        original_points[:,:,2] = 1  # we reuse 3rd column in completely different way here, it is hack
        converted_points = np.matmul(M, original_points.transpose([0,2,1])).transpose([0,2,1])
        meta['keypoints'][:,:,0:2] = converted_points

        # we just made image flip, i.e. right leg just became left leg, and vice versa
        if self.flip:
            tmpLeft = meta['keypoints'][:, self.leftParts, :]
            tmpRight = meta['keypoints'][:, self.rightParts, :]
            meta['keypoints'][:, self.leftParts, :] = tmpRight
            meta['keypoints'][:, self.rightParts, :] = tmpLeft

        return image, mask, meta
