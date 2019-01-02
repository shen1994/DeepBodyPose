# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:21:51 2018

@author: shen1994
"""

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
    model.load_weights("model/weights.0100.h5")
    
    transfer = Utils()
    
    # camera settins
    vedio_shape = [1920, 1080]
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, vedio_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vedio_shape[1])
    cv2.namedWindow("Deep3DPose", cv2.WINDOW_NORMAL)

    while(True):
        
        _, o_image = cap.read()
        
        r_image = cv2.resize(o_image, (368, 368))

        output_blobs = model.predict(np.array([r_image]))
            
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, (r_image.shape[1], r_image.shape[0]), interpolation=cv2.INTER_CUBIC)
            
        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(paf, (r_image.shape[1], r_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        points_image = transfer.transfer_to_points(o_image, heatmap)
        # lines_image = transfer.transfer_to_lines(o_image, heatmap, paf)

        cv2.imshow("Deep3DPose", points_image)
            
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        
    cv2.destroyAllWindows()
    

        