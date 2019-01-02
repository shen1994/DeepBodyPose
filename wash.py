# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:14:15 2018

@author: shen1994
"""

import os
import sys
sys.path.append('images/cocoapi/PythonAPI')
import numpy as np
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist

def generate_info(anno_dir, image_dir, mask_dir, dataset_mode="train"):
    
    dataset_count = 0
    all_joints = []

    coco = COCO(anno_dir)
    ids = list(coco.imgs.keys())
        
    for img_i, img_id in enumerate(ids):
            
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_ans = coco.loadAnns(ann_ids)
            
        # save properties
        all_persons = []
        for p in range(len(img_ans)):
                
            pers = {}
            per_center = [img_ans[p]["bbox"][0] + img_ans[p]["bbox"][2] / 2,
                          img_ans[p]["bbox"][1] + img_ans[p]["bbox"][3] / 2]
            pers["objpos"] = per_center
            pers["bbox"] = img_ans[p]["bbox"]
            pers["segment_area"] = img_ans[p]["area"]
            pers["num_keypoints"] = img_ans[p]["num_keypoints"]
                
            anno = img_ans[p]["keypoints"]
            pers["keypoints"] = np.zeros((17, 3))            
            for part in range(17):
                pers["keypoints"][part, 0] = anno[part * 3]
                pers["keypoints"][part, 1] = anno[part * 3 + 1]

                if anno[part * 3 + 2] == 2:
                    pers["keypoints"][part, 2] = 1
                elif anno[part * 3 + 2] == 1:
                    pers["keypoints"][part, 2] = 0
                else:
                    pers["keypoints"][part, 2] = 2

            pers["scale_provided"] = img_ans[p]["bbox"][3] / 368.
  
            all_persons.append(pers)
            
        # discard useless properties
        prev_centers = []
        main_persons = []
        for pers in all_persons:
                
            if pers["num_keypoints"] < 5 or pers["segment_area"] < 32 * 32:
                continue
                
            flag = False
            for pc in prev_centers:
                a = np.expand_dims(pc[:2], axis=0)
                b = np.expand_dims(pers["objpos"], axis=0)
                dist = cdist(a, b)[0]
                if dist < pc[2] * 0.3:
                    flag = True
                    continue
                
            if flag:
                continue
                
            main_persons.append(pers)
            prev_centers.append(np.append(pers["objpos"], \
                                          max(img_ans[p]["bbox"][2], img_ans[p]["bbox"][3])))
        
        # main persons is more, whose samples is more
        # such as the number of main persons is 5, we will generate 5 samples
        for p, person in enumerate(main_persons):
            all_joints.append(dict())

            all_joints[dataset_count]["image_id"] = img_id
            all_joints[dataset_count]["image_path"] = os.path.join(image_dir, 'COCO_%s2014_%012d.jpg' %(dataset_mode, img_id))
            all_joints[dataset_count]["mask_miss_path"] = os.path.join(mask_dir, 'mask_miss_%012d.png' % img_id)
            all_joints[dataset_count]["mask_all_path"] = os.path.join(mask_dir, 'mask_all_%012d.png' % img_id)
            
            all_joints[dataset_count]["keypoints"] = []
            all_joints[dataset_count]["num_keypoints"] = []
            all_joints[dataset_count]["objpos"] = []
            all_joints[dataset_count]["bbox"] = []
            all_joints[dataset_count]["scale_provided"] = []
                            
            all_joints[dataset_count]["keypoints"].append(main_persons[p]["keypoints"])
            all_joints[dataset_count]["num_keypoints"].append(main_persons[p]["num_keypoints"])
            all_joints[dataset_count]["objpos"].append(main_persons[p]["objpos"])
            all_joints[dataset_count]["bbox"].append(main_persons[p]["bbox"])
            all_joints[dataset_count]["scale_provided"].append(main_persons[p]["scale_provided"])

            for op, operson in enumerate(all_persons):
                if person is operson:
                    continue
                if operson["num_keypoints"] == 0:
                    continue
                
                all_joints[dataset_count]["keypoints"].append(all_persons[op]["keypoints"])
                all_joints[dataset_count]["num_keypoints"].append(all_persons[op]["num_keypoints"])
                all_joints[dataset_count]["objpos"].append(all_persons[op]["objpos"])
                all_joints[dataset_count]["bbox"].append(all_persons[op]["bbox"])
                all_joints[dataset_count]["scale_provided"].append(all_persons[op]["scale_provided"])
                            
            dataset_count += 1
                
    return all_joints
    

if __name__ == "__main__":
    
    dataset_dir = 'images/'
    
    # train
    tra_anno_dir = os.path.join(dataset_dir, 'annotations/person_keypoints_train2014.json')
    tra_image_dir = os.path.join(dataset_dir, 'train2014')
    tra_mask_dir = os.path.join(dataset_dir, 'trainmask2014')
    
    info = generate_info(tra_anno_dir, tra_image_dir, tra_mask_dir, "train")

    tra_npy_dir = os.path.join(dataset_dir, "train_dataset_2014.npy")
    np.save(tra_npy_dir, info)
    
    # valid
    val_anno_dir = os.path.join(dataset_dir, 'annotations/person_keypoints_val2014.json')
    val_image_dir = os.path.join(dataset_dir, 'val2014')
    val_mask_dir = os.path.join(dataset_dir, 'valmask2014')
    
    info = generate_info(val_anno_dir, val_image_dir, val_mask_dir, "val")

    val_npy_dir = os.path.join(dataset_dir, "valid_dataset_2014.npy")
    np.save(val_npy_dir, info)      
