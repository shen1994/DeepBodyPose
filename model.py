# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:26:55 2018

@author: shen1994
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers.merge import Concatenate

from blocks import vgg_block
from blocks import stage1_block
from blocks import stageT_block
from blocks import mask_block

def train_pose_model(input_shape=(368, 368, 3), stride=8, stages=6, weight_decay=None):
    
    image_input = Input(shape=input_shape, name='train_pose_input')
    vect_input = Input(shape=(input_shape[0]//stride, input_shape[1]//stride, 38))
    heat_input = Input(shape=(input_shape[0]//stride, input_shape[1]//stride, 19))
    
    image_norm = Lambda(lambda x: x / 256. - 0.5)(image_input)
    
    outputs = []
    
    # vgg
    stage0_out = vgg_block(image_norm, weight_decay)
    
    # stage 1 - branch 1 (PAFs)
    stage1_branch1_out = stage1_block(stage0_out, 38, 1, weight_decay)
    w1 = mask_block(stage1_branch1_out, vect_input, heat_input, 38, 1, 1)
    
    # stage 1 - branch 2 (configdence maps)
    stage1_branch2_out = stage1_block(stage0_out, 19, 2, weight_decay)
    w2 = mask_block(stage1_branch2_out, vect_input, heat_input, 19, 1, 2)
    
    outputs.append(w1)
    outputs.append(w2)
    
    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])
    
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAFs)
        stageT_branch1_out = stageT_block(x, 38, sn, 1, weight_decay)
        w1 = mask_block(stageT_branch1_out, vect_input, heat_input, 38, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, 19, sn, 2, weight_decay)
        w2 = mask_block(stageT_branch2_out, vect_input, heat_input, 19, sn, 2)
        
        outputs.append(w1)
        outputs.append(w2)
        
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])
        
    return Model(inputs=[image_input, vect_input, heat_input], outputs=outputs)
    
def pose_model(input_shape=(368, 368, 3), stages=6):   
    
    image_input = Input(shape=input_shape, name='pose_input')

    image_morm = Lambda(lambda x: x / 256. - 0.5)(image_input)

    # VGG
    stage0_out = vgg_block(image_morm, None)

    # stage 1 - branch 1 (PAFs)
    stage1_branch1_out = stage1_block(stage0_out, 38, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, 19, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, 38, sn, 1, None)
        stageT_branch2_out = stageT_block(x, 19, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    return Model(inputs=[image_input], outputs=[stageT_branch1_out, stageT_branch2_out])
    