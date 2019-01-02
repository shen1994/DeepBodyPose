# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:14:24 2018

@author: shen1994
"""

import cv2
import numpy as np
from skimage.transform import resize

def get_transpose_axes(n):
    
    if n % 2 == 0:
        y_axes = list(range( 1, n-1, 2 ))
        x_axes = list(range( 0, n-1, 2 ))
    else:
        y_axes = list(range( 0, n-1, 2 ))
        x_axes = list(range( 1, n-1, 2 ))
        
    return y_axes, x_axes, [n-1]

def stack_images(images):
    
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len( images_shape ))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]

    return np.transpose(images, axes=np.concatenate(new_axes)).reshape( new_shape)
    
def resize_images(images, size):

    new_images = []
    for image in images:
        nimage = resize(image, (size, size), preserve_range = True)
        new_images.append(nimage)
    return np.array(new_images)

def show_G(images_A, images_B, batch_size, name):
    
    images_A = resize_images(images_A, 128)
    images_B = resize_images(images_B, 128)
    figure = np.stack([images_A, images_B], axis=1)
    figure = figure.reshape((4, batch_size//4) + figure.shape[1:])
    figure = stack_images(figure).astype(np.uint8)

    cv2.imshow(name, figure) 
    cv2.waitKey(1)
    
def show_predictor(images, model, transfer, batch_size, name):
    
    images_A, images_B = [], []
    for index in range(batch_size):   
        image = images[index] 
        canvas = image.copy()
      
        output_blobs = model.predict(np.array([image]))
        
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        points_image = transfer.transfer_to_points(canvas, heatmap)
                            
        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(paf, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_CUBIC)
                        
        lines_image = transfer.transfer_to_lines(canvas, heatmap, paf)
        
        images_A.append(points_image)
        images_B.append(lines_image)
        
    show_G(images_A, images_B, batch_size, name)
                