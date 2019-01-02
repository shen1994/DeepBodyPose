# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:30:14 2018

@author: shen1994
"""

import cv2
import numpy as np
from argu import Argument
from heatmap import HeatMapper

class Generator:
    
    def __init__(self, 
                 path):
        
        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
        self.coco_parts = ["nose", "Leye", "Reye", "Lear", "Rear", "Lsho", "Rsho", "Lelb", "Relb", "Lwri", "Rwri", "Lhip", "Rhip", "Lkne", "Rkne", "Lank", "Rank"]
        self.num_parts = len(self.parts)
        self.num_coco_parts = len(self.coco_parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.coco_parts_dict = dict(zip(self.coco_parts, range(self.num_coco_parts)))

        self.vec_num = 38
        self.heat_num = 19

        self.samples = np.load(path)
        self.samples_length = len(self.samples)
        self.argument = Argument()
        self.heatmapper = HeatMapper()
        
    def convert(self, joints):
        
        result = np.zeros((joints.shape[0], self.num_parts, 3), dtype=np.float)
        result[:,:,2]=2.  # 2 - abstent, 1 visible, 0 - invisible

        neckG = self.parts_dict['neck']
        RshoC = self.parts_dict['Rsho']
        LshoC = self.parts_dict['Lsho']
   
        for p in self.parts:
            global_id = self.parts_dict[p]
            if global_id == self.parts_dict["neck"]:
                continue
            coco_id = self.coco_parts_dict[p]
            result[:,global_id,:]=joints[:,coco_id,:]

        # no neck in coco database, we calculate it as averahe of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2]<2)

        result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                    joints[both_shoulders_known, LshoC, 0:2]) / 2
        result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                            joints[both_shoulders_known, LshoC, 2])
  
        return result
        
    def show_points(self):
        
        np.random.shuffle(self.samples)
        
        meta = self.samples[0].copy()
        meta["keypoints"] = self.convert(np.array(meta["keypoints"]))
        
        image = cv2.imread(meta["image_path"], 1)
        mask = cv2.imread(meta["mask_miss_path"], 0)
        image, mask, meta = self.argument.transform(image, mask, meta, is_random=True)
        while(True):
            for i in range(len(meta["keypoints"])): 
                one = meta["keypoints"][i]
                for ione in one:
                    cv2.circle(image, (int(ione[0]), int(ione[1])), 5, (0, 0, 255), -1)
            cv2.imshow("d", image)
            if cv2.waitKey(1) == ord('q'):
                break
            
    def show_heatmapper(self):
        
        np.random.shuffle(self.samples)
        
        meta = self.samples[0].copy()
        meta["keypoints"] = self.convert(np.array(meta["keypoints"]))
 
        image = cv2.imread(meta["image_path"], 1)
        mask = cv2.imread(meta["mask_miss_path"], 0)
        image, mask, meta = self.argument.transform(image, mask, meta, is_random=True)
        paf = self.heatmapper.create_heatmaps(meta['keypoints']) * mask
    
        parts = []
        heat_pattern = np.zeros((368, 368, 3), dtype=np.uint8)
        for i in range(18):
            heated_image = image.copy()
            heat_img = paf[38+i]
            heat_img = cv2.resize(heat_img, (368, 368), interpolation=cv2.INTER_NEAREST)
            heat_img = heat_img.reshape((368, 368, 1))
            heated_image = heated_image*(1-heat_img) + heat_pattern*heat_img    
            parts += [heated_image]
        parts = np.vstack(parts)
        cv2.imwrite("heats.png", parts)
    
        limb_from = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
        limb_to = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        limbs_conn = zip(limb_from, limb_to)
        pafs = []
        for i,(fr,to) in enumerate(limbs_conn):
            paffed_image = image.copy()
            pafX = paf[i * 2]
            pafY = paf[i * 2 + 1]
            for x in range(368//8):
                for y in range(368//8):
                    X = pafX[y, x]
                    Y = pafY[y, x]
                    if X!=0 or Y!=0:
                        cv2.arrowedLine(paffed_image, (x*8,y*8), (int(x*8+X*8),int(y*8+Y*8)), 
                                        color=(0,0,255), thickness=1, tipLength=0.5)
            pafs += [paffed_image]
        pafs = np.vstack(pafs)
        cv2.imwrite("pafs.png", pafs)        
    
    def generate(self, batch_size=32, is_random=True):
        
        while(True):
         
            np.random.shuffle(self.samples)
            
            counter = 0
            batch_x, batch_x1, batch_x2 = [], [], []
            batch_y1, batch_y2 = [], [] 
            for i in range(self.samples_length):
                
                meta = self.samples[i].copy()
                meta["keypoints"] = self.convert(np.array(meta["keypoints"]))
                image = cv2.imread(meta["image_path"], 1)
    
                mask = cv2.imread(meta["mask_miss_path"], 0)
                image, mask, meta = self.argument.transform(image, mask, meta, is_random=is_random)
                label = self.heatmapper.create_heatmaps(meta['keypoints']) * mask
    
                batch_x.append(image)
                vec_weights = np.repeat(mask[:,:,np.newaxis], self.vec_num, axis=2)
                heat_weights = np.repeat(mask[:,:,np.newaxis], self.heat_num, axis=2)
                batch_x1.append(vec_weights)
                batch_x2.append(heat_weights)
                
                vec_label = label[:self.vec_num, :, :]
                vec_label = np.transpose(vec_label, (1, 2, 0))
                heat_label = label[self.vec_num:self.vec_num+self.heat_num, :, :]
                heat_label = np.transpose(heat_label, (1, 2, 0))
                batch_y1.append(vec_label)
                batch_y2.append(heat_label)
                
                counter += 1
                
                if counter >= batch_size:
                    
                    yield [np.array(batch_x), np.array(batch_x1), np.array(batch_x2)], \
                          [np.array(batch_y1), np.array(batch_y2), 
                           np.array(batch_y1), np.array(batch_y2),
                           np.array(batch_y1), np.array(batch_y2),
                           np.array(batch_y1), np.array(batch_y2),
                           np.array(batch_y1), np.array(batch_y2),
                           np.array(batch_y1), np.array(batch_y2)]
                    counter = 0
                    batch_x, batch_x1, batch_x2 = [], [], []
                    batch_y1, batch_y2 = [], []
       
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import pylab
    generator = Generator("images/train_dataset_2014.npy")
    # generator.show_points()
    # generator.show_heatmapper()

    x, y = generator.generate(batch_size=3, is_random=True).__next__()
    
    batch_index = 0
    
    body_part = 0
    dta_img = x[0][batch_index,:,:,:]
    plt.imshow(dta_img[:,:,[2,1,0]])   
    heatmap = cv2.resize(y[1][batch_index, :, :, body_part], (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    plt.imshow(dta_img[:,:,[2,1,0]])
    plt.imshow(heatmap[:,:], alpha=.5)
    pylab.show()
    
    paf_idx = 22   
    paf = cv2.resize(y[0][batch_index, :, :, paf_idx], (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)   
    plt.imshow(dta_img[:,:,[2,1,0]])
    plt.imshow(paf[:,:], alpha=.5)
    pylab.show()
    
    mask_img = x[1][batch_index,:,:,:]
    mask_img = cv2.resize(mask_img[:,:, 0], (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    plt.imshow(dta_img[:,:,[2,1,0]])
    plt.imshow(mask_img * 255, cmap='gray', alpha=.5)
    pylab.show()
    '''
    X, Y = generator.generate(batch_size=1, is_random=True).__next__()
    print(X[0].shape, X[1].shape, X[2].shape)
    print(Y[0].shape, Y[1].shape)    
    X, Y = generator.generate(batch_size=1).__next__()
    print(X[0].shape, X[1].shape, X[2].shape)
    print(Y[0].shape, Y[1].shape)  
    '''
            