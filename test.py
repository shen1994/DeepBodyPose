# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:47:09 2018

@author: shen1994
"""

import os
import cv2
import numpy as np
from model import pose_model

from utils import Utils

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    model = pose_model(input_shape=(None, None, 3), stages=6)
    model.load_weights("model/weights.84-3000-91.37.hdf5")
    # model.load_weights("model/weights.0100.h5")
    
    transfer = Utils()
    
    image = cv2.imread("test.jpg")
    image_h, image_w = image.shape[0], image.shape[1]
    r_image = cv2.resize(image, (368, 368))
    
    input_img = np.array([r_image])
    
    output_blobs = model.predict(input_img)
        
    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.resize(heatmap, (r_image.shape[1], r_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # points_image = transfer.transfer_to_points(image, heatmap)
        
    paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
    paf = cv2.resize(paf, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (r_image.shape[1], r_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    lines_image = transfer.transfer_to_lines(image, heatmap, paf)
    
    # rects_image = transfer.transfer_to_rects(image, heatmap, paf)
    
    while(True):
        
        cv2.imshow("Deep2DPose", lines_image)
        if cv2.waitKey(1) == ord('q'):
            break
            