# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:27:52 2018

@author: shen1994
"""

import os
import cv2
import sys
sys.path.append('images/cocoapi/PythonAPI')
import numpy as np
from pycocotools.coco import COCO


def preprocess(mode):
    
    dataset_dir = 'images/'
    annos_dir = os.path.join(dataset_dir, 'annotations/person_keypoints_%s2014.json'%mode)
    images_dir = os.path.join(dataset_dir, '%s2014' %mode)
    masks_dir = os.path.join(dataset_dir, '%smask2014' %mode)
    if not os.path.exists(masks_dir):
        os.mkdir(masks_dir)
    
    coco = COCO(annos_dir)
    ids = list(coco.imgs.keys())
    for i, i_id in enumerate(ids):
        ann_ids = coco.getAnnIds(imgIds=i_id)
        img_ans = coco.loadAnns(ann_ids)
        
        img_path = os.path.join(images_dir, "COCO_%s2014_%012d.jpg" %(mode, i_id))
        mask_miss_path = os.path.join(masks_dir, "mask_miss_%012d.png" %i_id)
        mask_all_path = os.path.join(masks_dir, "mask_all_%012d.png" %i_id)
        
        img = cv2.imread(img_path)
        h, w, c = img.shape
        mask_all = np.zeros((h, w), dtype=np.uint8)
        mask_miss = np.zeros((h, w), dtype=np.uint8)
        flag = 0
        for p in img_ans:
            if p["iscrowd"] == 1:
                mask_crowd = coco.annToMask(p)
                temp = np.bitwise_and(mask_all, mask_crowd)
                mask_crowd = mask_crowd - temp
                flag += 1
                continue
            else:
                mask = coco.annToMask(p)
                
            mask_all = np.bitwise_or(mask, mask_all)

            if p["num_keypoints"] <= 0:
                mask_miss = np.bitwise_or(mask, mask_miss)  
                
        if flag<1:
            mask_miss = np.logical_not(mask_miss)
        elif flag == 1:
            mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
            mask_all = np.bitwise_or(mask_all, mask_crowd)
        else:
            raise Exception("crowd segments > 1")

        cv2.imwrite(mask_miss_path, mask_miss * 255)
        cv2.imwrite(mask_all_path, mask_all * 255)

        if (i % 1000 == 0):
            print("Processed %d of %d" % (i, len(ids)))

    print("Done !!!")

if __name__ == "__main__":
    
    preprocess('train') 
    preprocess('val')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  