# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:03:28 2018

@author: shen1994
"""

import re
import os
import cv2
import keras
from keras import backend as K
from model import train_pose_model
from model import pose_model
from generate import Generator
from optimizers import MultiSGD
from utils import Utils
from show import show_predictor

from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19

def load_vgg_weights(model):
    
    from_vgg = dict()
    from_vgg['conv1_1'] = 'block1_conv1'
    from_vgg['conv1_2'] = 'block1_conv2'
    from_vgg['conv2_1'] = 'block2_conv1'
    from_vgg['conv2_2'] = 'block2_conv2'
    from_vgg['conv3_1'] = 'block3_conv1'
    from_vgg['conv3_2'] = 'block3_conv2'
    from_vgg['conv3_3'] = 'block3_conv3'
    from_vgg['conv3_4'] = 'block3_conv4'
    from_vgg['conv4_1'] = 'block4_conv1'
    from_vgg['conv4_2'] = 'block4_conv2'
    
    vgg_model = VGG19(include_top=False, weights='imagenet')
    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
    
def optimizer_lr_mult(model):
    
    lr_mult=dict()
    for layer in model.layers:  
        if isinstance(layer, Conv2D):
            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2
            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8
            # vgg
            else:
               kernel_name = layer.weights[0].name
               bias_name = layer.weights[1].name
               lr_mult[kernel_name] = 1
               lr_mult[bias_name] = 2
  
    return lr_mult  
        
def total_eucl_loss(in_, out_, num, batch_size=16):
    
    loss = []
    for i in range(num):
        loss.append(K.sum(K.square(out_[i] - in_[i])) / batch_size)

    return K.sum(loss) / num

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    model_path = 'model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
      
    # define parameters
    image_shape = (368, 368, 3)
    epochs = 1000
    batch_size = 16
    
    # define model
    model = train_pose_model(input_shape=image_shape)
    try:
        model.load_weights('model/weights.09-149.61.hdf5', by_name=True)
    except Exception:
        load_vgg_weights(model)
    
    # define transfer
    transfer = Utils()
    test_model = pose_model(input_shape=(None, None, 3), stages=6)
    
    # define train_model
    labels = [Input(shape=(image_shape[0]//8, image_shape[1]//8, 38)), Input(shape=(image_shape[0]//8, image_shape[1]//8, 19)),
              Input(shape=(image_shape[0]//8, image_shape[1]//8, 38)), Input(shape=(image_shape[0]//8, image_shape[1]//8, 19)),
              Input(shape=(image_shape[0]//8, image_shape[1]//8, 38)), Input(shape=(image_shape[0]//8, image_shape[1]//8, 19)),
              Input(shape=(image_shape[0]//8, image_shape[1]//8, 38)), Input(shape=(image_shape[0]//8, image_shape[1]//8, 19)),
              Input(shape=(image_shape[0]//8, image_shape[1]//8, 38)), Input(shape=(image_shape[0]//8, image_shape[1]//8, 19)),
              Input(shape=(image_shape[0]//8, image_shape[1]//8, 38)), Input(shape=(image_shape[0]//8, image_shape[1]//8, 19))]
    params = model.trainable_weights
    loss = total_eucl_loss(model.output, labels, len(labels), batch_size=batch_size)
    # opt = keras.optimizers.SGD(lr=1e-5, momentum=0.9, decay=1e-6, nesterov=True).get_updates(params, [], loss)    
    opt = MultiSGD(lr=1e-5, momentum=0.9, decay=1e-6, nesterov=True, lr_mult=optimizer_lr_mult(model)).get_updates(params, [], loss)    
    train_model = K.function(model.input + labels + [K.learning_phase()], [loss], opt)

    # define generaters
    t_generator = Generator("images/train_dataset_2014.npy")
    v_generator = Generator("images/valid_dataset_2014.npy")
    
    # train for model
    for epoch in range(epochs):
        t_total_coss = 0
        t_steps = t_generator.samples_length // batch_size
        for step in range(t_steps):
            
            # train
            X, Y = t_generator.generate(batch_size=batch_size, is_random=True).__next__()
            t_coss = train_model(X + Y)[0]
            t_total_coss += t_coss
            
            # watch
            if step % 100 == 0:
                
                # 1. show coss
                X, Y = v_generator.generate(batch_size=batch_size, is_random=False).__next__()
                v_coss = train_model(X + Y + [1.])[0]
                print("steps: %d, step: %d, t_coss: %.2f, v_coss: %.2f" %(t_steps, step, t_total_coss, v_coss))
                
                # 2. show test images 
                model.save_weights("model/weights.%02d-%02d-%.2f.hdf5" %(epoch, step, v_coss))
                test_model.load_weights("model/weights.%02d-%02d-%.2f.hdf5" %(epoch, step, v_coss))
                show_predictor(X[0], test_model, transfer, batch_size//2, "Deep2DPose")
                
                t_total_coss = 0
                
        # save model
        X, Y = v_generator.generate(batch_size=batch_size, is_random=False).__next__()
        v_coss = train_model(X + Y + [1.])[0]
        model.save_weights("model/weights.%02d-%.2f.hdf5" %(epoch, v_coss))
                
        # exit programs
        if cv2.waitKey(1) == ord('q'):
            exit()
