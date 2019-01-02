# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:14:15 2018

@author: shen1994
"""

import os
import cv2
import sys
import h5py
import json
import struct
sys.path.append('images/cocoapi/PythonAPI')
import numpy as np
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist

def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    if type(floats) is int:
        floats = [float(floats)]
    if type(floats) is list and \
        len(floats) > 0 and \
        type(floats[0]) is list:
        floats = floats[0]
    return struct.pack('%sf' % len(floats), *floats)

if __name__ == "__main__":
    
    dataset_dir = 'images/'
    tra_anno_dir = os.path.join(dataset_dir, 'annotations/person_keypoints_train2014.json')
    tra_image_dir = os.path.join(dataset_dir, 'train2014')
    tra_mask_dir = os.path.join(dataset_dir, 'trainmask2014')
    
    val_anno_dir = os.path.join(dataset_dir, 'annotations/person_keypoints_val2014.json')
    val_image_dir = os.path.join(dataset_dir, 'val2014')
    val_mask_dir = os.path.join(dataset_dir, 'valmask2014')
    
    datasets = [
        (val_anno_dir, val_image_dir, val_mask_dir, "COCO_val", "val"),
        (tra_anno_dir, tra_image_dir, tra_mask_dir, "COCO_tra", "tra")
    ]
    
    tra_hdf5_dir = os.path.join(dataset_dir, "train_dataset_2014.h5")
    val_hdf5_dir = os.path.join(dataset_dir, "val_dataset_2014.h5")
    
    dataset_count = 0
    all_joints = []
    
    for _, ds in enumerate(datasets):
        
        anno_dir = ds[0]
        image_dir = ds[1]
        mask_dir = ds[2]
        dataset_type = ds[3]
        dataset_mode = ds[4]

        coco = COCO(anno_dir)
        ids = list(coco.imgs.keys())
        
        for img_i, img_id in enumerate(ids):
            
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_ans = coco.loadAnns(ann_ids)
    
            image = coco.imgs[img_id]
            h, w = image["height"], image["width"]

            # save properties
            all_persons = []
            for p in range(len(img_ans)):
                
                pers = {}
                per_center = [img_ans[p]["bbox"][0] + img_ans[p]["bbox"][2] / 2,
                              img_ans[p]["bbox"][1] + img_ans[p]["bbox"][3] / 2]

                pers["object_position"] = per_center
                pers["object_bbox"] = img_ans[p]["bbox"]
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

                pers["scale_provided"] = img_ans[p]["bbox"][3] / 368

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
                    b = np.expand_dims(pers["object_position"], axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2] * 0.3:
                        flag = True
                if flag:
                    continue
                
                main_persons.append(pers)
                prev_centers.append(np.append(pers["object_position"], \
                                              max(img_ans[p]["bbox"][2], img_ans[p]["bbox"][3])))
                
            for p, person in enumerate(main_persons):
                all_joints.append(dict())
                all_joints[dataset_count]["dataset_type"] = dataset_type

                if img_i < 2645 and "val" in dataset_type:
                    isValidation = 1
                else:
                    isValidation = 0
                all_joints[dataset_count]["is_validation"] = isValidation
                
                all_joints[dataset_count]["image_height"] = h
                all_joints[dataset_count]["image_width"] = w
                all_joints[dataset_count]["image_id"] = img_id
                all_joints[dataset_count]["image_index"] = img_i
                all_joints[dataset_count]["image_path"] = os.path.join(image_dir, 'COCO_%s2014_%012d.jpg' %(dataset_mode, img_id))
                all_joints[dataset_count]["mask_miss_path"] = os.path.join(mask_dir, 'mask_miss_%012d.png' % img_id)
                all_joints[dataset_count]["mask_all_path"] = os.path.join(mask_dir, 'mask_all_%012d.png' % img_id)
                
                all_joints[dataset_count]["object_position"] = main_persons[p]["object_position"]
                all_joints[dataset_count]["object_bbox"] = main_persons[p]["object_bbox"]
                all_joints[dataset_count]["segment_area"] = main_persons[p]["segment_area"]
                all_joints[dataset_count]["num_keypoints"] = main_persons[p]["num_keypoints"]
                all_joints[dataset_count]["keypoints"] = main_persons[p]["keypoints"]
                all_joints[dataset_count]["scale_provided"] = main_persons[p]["scale_provided"]
                
                all_joints[dataset_count]["other_keypoints"] = []
                all_joints[dataset_count]["other_scale_provided"] = []
                all_joints[dataset_count]["other_object_position"] = []
                all_joints[dataset_count]["other_object_bbox"] = []
                all_joints[dataset_count]["other_segment_area"] = []
                all_joints[dataset_count]["other_num_keypoints"] = []

                other_number = 0
                for o, operson in enumerate(all_persons):
                    if person is operson:
                        all_joints[dataset_count]["people_index"] = o
                        continue
                    if operson["num_keypoints"] == 0:
                        continue
                    all_joints[dataset_count]["other_keypoints"].append(all_persons[o]["keypoints"])
                    all_joints[dataset_count]["other_scale_provided"].append(all_persons[o]["scale_provided"])
                    all_joints[dataset_count]["other_object_position"].append(all_persons[o]["object_position"])
                    all_joints[dataset_count]["other_object_bbox"].append(all_persons[o]["object_bbox"])
                    all_joints[dataset_count]["other_segment_area"].append(all_persons[o]["segment_area"])
                    all_joints[dataset_count]["other_num_keypoints"].append(all_persons[o]["num_keypoints"])
                    other_number += 1
                all_joints[dataset_count]["people_other_number"] = other_number

                dataset_count += 1

    # write to hdf5
    tra_h5 = h5py.File(tra_hdf5_dir, 'w')
    tra_gp = tra_h5.create_group('datum')
    tra_write_count = 0
    
    val_h5 = h5py.File(val_hdf5_dir, 'w')
    val_gp = val_h5.create_group('dataum')
    val_write_count = 0
    
    data = all_joints
    data_number = len(data)
    
    is_validation_array = [data[i]['is_validation'] for i in range(data_number)]
    val_total_write_count = is_validation_array.count(0.0)
    tra_total_write_count = len(data) - val_total_write_count

    for idx in range(data_number):
        image_path = data[idx]["image_path"]
        mask_all_path = data[idx]["mask_all_path"]
        mask_miss_path = data[idx]["mask_miss_path"]

        image = cv2.imread(image_path)
        mask_all = cv2.imread(mask_all_path, 0)
        mask_miss = cv2.imread(mask_miss_path, 0)
        
        h, w = image.shape[0], image.shape[1]
        if w < 64:
            image = cv2.copyMakeBorder(image, 0, 0, 0, 64-w, 
                                       cv2.BORDER_CONSTANT, value=(128, 128, 128))
            cv2.imwrite('padded_image.jpg', image)
            w = 64
        meta_data = np.zeros(shape=(h, w, 1), dtype=np.uint8)
        
        serializable_meta = {}

        clidx = 0
        for i in range(len(data[idx]['dataset_type'])):
            meta_data[clidx][i] = ord(data[idx]['dataset_type'][i])
        clidx += 1
        serializable_meta['dataset_type'] = data[idx]['dataset_type']

        for i in range(len(float2bytes(data[idx]['image_height']))):
            serializable_meta['image_height'] = data[idx]['image_height']
            serializable_meta['image_width'] = data[idx]['image_width']

        # (a) is_validation(uint8), people_other_number(uint8), people_index(uint8)
        # (a) anno_index(float), write_count(float), total_write_count(float)
        meta_data[clidx][0] = data[idx]['is_validation']
        meta_data[clidx][1] = data[idx]['people_other_number']
        meta_data[clidx][2] = data[idx]['people_index']
        anno_index_binary = float2bytes(data[idx]['image_index'])
        for i in range(len(anno_index_binary)):
            meta_data[clidx][3+i] = anno_index_binary[i]
        if data[idx]['is_validation']:
            count_binary = float2bytes(float(val_write_count))
        else:
            count_binary = float2bytes(float(tra_write_count))
        for i in range(len(count_binary)):
            meta_data[clidx][7+i] = count_binary[i]
        if data[idx]['is_validation']:
            total_write_count_binary = float2bytes(float(val_total_write_count))
        else:
            total_write_count_binary = float2bytes(float(tra_total_write_count))
        for i in range(len(total_write_count_binary)):
            meta_data[clidx][11+i] = total_write_count_binary[i]
        clidx += 1
        serializable_meta['is_validation'] = data[idx]['is_validation']
        serializable_meta['people_other_number'] = data[idx]['people_other_number']
        serializable_meta['people_index'] = data[idx]['image_index']
        serializable_meta['count'] = val_write_count if data[idx]['is_validation'] else tra_write_count
        serializable_meta['total_count'] = val_total_write_count if data[idx]['is_validation'] else tra_total_write_count
        
        # (b) object_position_x(float), object_position_y(float)
        object_position_binary = float2bytes(data[idx]['object_position'])
        for i in range(len(object_position_binary)):
            meta_data[clidx][i] = object_position_binary[i]
        clidx += 1
        serializable_meta['object_position'] = [data[idx]['object_position']]

        # (c) scale_provided(float)
        scale_provided_binary = float2bytes(data[idx]['scale_provided'])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = scale_provided_binary[i]
        clidx += 1
        serializable_meta['scale_provided'] = [data[idx]['scale_provided']]
        
        # (d) self keypoints (3*16)(float)(3 line)
        keypoints = np.asarray(data[idx]['keypoints']).T.tolist()
        for i in range(len(keypoints)):
            row_binary = float2bytes(keypoints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = row_binary[j]
            clidx += 1
        serializable_meta['keypoints'] = [data[idx]['keypoints'].tolist()]

        # (e) check nop
        nop = int(data[idx]['people_other_number'])
        if (not nop == 0):
            other_keypoints = data[idx]['other_keypoints']
            other_object_position = data[idx]['other_object_position']
            other_scale_provided = data[idx]['other_scale_provided']

            # (f) other_object_position_x(float), other_object_position_y(float) (nop lines)
            for i in range(nop):
                object_position_binary = float2bytes(other_object_position[i])
                for j in range(len(object_position_binary)):
                    meta_data[clidx][j] = object_position_binary[j]
                clidx += 1
            serializable_meta['object_position'].extend(data[idx]['other_object_position'])
            
            # (g) other_scale_provided(float)
            other_scale_provided_binary = float2bytes(other_scale_provided)
            for j in range(len(other_scale_provided_binary)):
                meta_data[clidx][j] = other_scale_provided_binary[j]
            clidx += 1
            serializable_meta['scale_provided'].extend(data[idx]['other_scale_provided'])
            
            # (h) other_keypoints (3*16)(float)(nop*3*lines)
            for n in range(nop):
                keypoints = np.asarray(other_keypoints[n]).T.tolist()
                for i in range(len(keypoints)):
                    row_binary = float2bytes(keypoints[i])
                    for j in range(len(row_binary)):
                        meta_data[clidx][j] = row_binary[j]
                    clidx += 1
                serializable_meta['keypoints'].append(other_keypoints[n].tolist())
                
        
        # (i) image_path, mask_all_path, mask_miss_path
        serializable_meta['image_path'] = image_path
        serializable_meta['mask_all_path'] = mask_all_path
        serializable_meta['mask_miss_path'] = mask_miss_path
        
        if "COCO" in data[idx]['dataset_type']:
            img4ch = np.concatenate((image, meta_data, mask_miss[..., None], \
                                     mask_all[..., None]), axis=2)
        elif "MPI" in data[idx]['dataset']:
            img4ch = np.concatenate((image, meta_data, mask_miss[..., None]), \
                                     axis=2)
        else:
            pass
        
        print(serializable_meta['keypoints'])
        print(len(serializable_meta['keypoints'][0]), len(serializable_meta['keypoints'][1]))
        print(len(serializable_meta['keypoints']))
        break
    
        
        if data[idx]['is_validation']:
            key = '%07d' % val_write_count
            ds = val_gp.create_dataset(key, data=img4ch, chunks=None)
            ds.attrs['meta'] = json.dumps(serializable_meta)
            val_write_count += 1
        else:
            key = '%07d' % tra_write_count
            ds = tra_gp.create_dataset(key, data=img4ch, chunks=None)
            ds.attrs['meta'] = json.dumps(serializable_meta)
            tra_write_count += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
