# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:25:54 2018

@author: shen1994
"""

import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class Utils:
    
    def __init__(self):
        # find connection in the specified sequence, center 29 is in the position 15
        self.limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                   [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                   [1,16], [16,18], [3,17], [6,18]]
                   
        # the middle joints heatmap correpondence
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
                  [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
                  [55,56], [37,38], [45,46]]
                  
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def get_peak_points(self, heatmap, threshold=0.06):
        
        all_peaks = []
        peak_counter = 0
        
        for part in range(18):
            map_ori = heatmap[:,:,part]
            map_gauss = gaussian_filter(map_ori, sigma=3)
            
            map_left = np.zeros(map_gauss.shape)
            map_left[1:,:] = map_gauss[:-1,:]
            map_right = np.zeros(map_gauss.shape)
            map_right[:-1,:] = map_gauss[1:,:]
            map_up = np.zeros(map_gauss.shape)
            map_up[:,1:] = map_gauss[:,:-1]
            map_down = np.zeros(map_gauss.shape)
            map_down[:,:-1] = map_gauss[:,1:]
            
            peaks_binary = np.logical_and.reduce((map_gauss>=map_left, map_gauss>=map_right, 
                                                  map_gauss>=map_up, map_gauss>=map_down, map_gauss > threshold))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
            
        return all_peaks
        
    def transfer_to_points(self, image, heatmap):
        
        image_w, image_h = image.shape[0], image.shape[1]
        canvas = image.copy()
        
        all_peaks = self.get_peak_points(heatmap)

        for i in range(18):
            for j in range(len(all_peaks[i])):
                x = int(all_peaks[i][j][0] / 368. * image_h)
                y = int(all_peaks[i][j][1] / 368. * image_w)
                cv2.circle(canvas, (x, y), 4, self.colors[i], thickness=-1)
                
        return canvas
        
    def get_connection_candidates(self, oriImg, paf, all_peaks, threshold=0.1):
        
        connection_all = []
        special_k = []
        mid_num = 10
        
        for k in range(len(self.mapIdx)):
            score_mid = paf[:,:,[x-19 for x in self.mapIdx[k]]]
            candA = all_peaks[self.limbSeq[k][0]-1]
            candB = all_peaks[self.limbSeq[k][1]-1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = self.limbSeq[k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)
                        
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
        
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > threshold)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
        
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0,5))
                for c in range(len(connection_candidate)):
                    i,j,s = connection_candidate[c][0:3]
                    if(i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break
        
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])     
        
        return connection_all, special_k
        
    def get_subsets(self, all_peaks, connection_all, special_k):
    
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        
        for k in range(len(self.mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(self.limbSeq[k]) - 1
        
                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    
                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
        
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])     
        
        return subset, candidate
        
    def transfer_to_lines(self, image, heatmap, paf, stickwidth=4):
        
        image_w, image_h = image.shape[0], image.shape[1]
        canvas = image.copy()
        
        all_peaks = self.get_peak_points(heatmap)
        
        connection_all, special_k = \
            self.get_connection_candidates(image, paf, all_peaks) 
            
        subset, candidate = \
            self.get_subsets(all_peaks, connection_all, special_k)
            
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(self.limbSeq[i])-1]
                if -1 in index:
                    continue
                Y = candidate[index.astype(int), 0] / 368. * image_h
                X = candidate[index.astype(int), 1] / 368. * image_w
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, self.colors[i])
                
        return canvas 
        
    def transfer_to_rects(self, image, heatmap, paf, stickwidth=4):
        
        image_w, image_h = image.shape[0], image.shape[1]
        canvas = image.copy()
        
        all_peaks = self.get_peak_points(heatmap)
        
        connection_all, special_k = \
            self.get_connection_candidates(image, paf, all_peaks) 
            
        subset, candidate = \
            self.get_subsets(all_peaks, connection_all, special_k)

        for n in range(len(subset)):
            X, Y = [], []
            for i in range(17):
                index = subset[n][np.array(self.limbSeq[i])-1]
                if -1 in index:
                    continue
                X += (candidate[index.astype(int), 0] / 368. * image_h).tolist()
                Y += (candidate[index.astype(int), 1] / 368. * image_w).tolist()
             
            if len(X) < 5:
                continue
            
            x_min, x_max = int(np.min(X)), int(np.max(X))
            y_min, y_max = int(np.min(Y)), int(np.max(Y)) 
            
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    
        return canvas        
        