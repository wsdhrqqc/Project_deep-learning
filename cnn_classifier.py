'''
Andrew H. Fagg

CNN-based classifiers

'''
#print('CNN Classifier')

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
#from tensorflow.keras.optimizers import RMSprop
#import random
import re

#from sklearn.p
import sklearn.metrics

def create_cnn_network(image_size, nchannels, 
                                  name_base='',
                                  conv_layers=[],                                  
                                  conv_activation='elu',
                                  dense_layers=[],
                                  dense_activation='elu',
                                  lrate=.0001,
                                  lambda_l2=None,
                                  p_dropout=None):
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels),
                        name='%sinput'%(name_base)))
    
    if lambda_l2 is not None:
        lambda_l2 = tf.keras.regularizers.l2(lambda_l2)
    
    # Loop over all convolutional layers
    i = 0
    for l in conv_layers:
        print("Layer", i, l)
        model.add(Convolution2D(filters=l['filters'],
                                kernel_size=l['kernel_size'],
                                strides=(1,1),
                                padding='valid',
                                use_bias=True,
                                kernel_initializer='truncated_normal',
                                bias_initializer='zeros',
                                name='%sC%d'%(name_base,i),
                                activation=conv_activation,
                                kernel_regularizer=lambda_l2))
        
        if l['pool_size'] is not None:
            model.add(MaxPooling2D(pool_size=l['pool_size'],
                               strides=l['strides'],
                               padding='valid'))
        i=i+1
        
    # Flatten 
    model.add(Flatten())
    
    # Loop over dense layers
    i = 0
    for l in dense_layers:
        model.add(Dense(units=l['units'],
                        activation=dense_activation,
                        use_bias=True,
                        kernel_initializer='truncated_normal',
                        bias_initializer='zeros',
                        name='%sD%d'%(name_base,i),
                        kernel_regularizer=lambda_l2))
        
        if p_dropout is not None:
            model.add(Dropout(p_dropout,
                              name='%sDR%d'%(name_base,i)))
            
        i=i+1
        
    return model
        
def create_cnn_classifier_network(image_size, nchannels, 
                                  name_base='',
                                  conv_layers=[],                                  
                                  conv_activation='elu',
                                  dense_layers=[],
                                  dense_activation='elu',
                                  n_classes=2, 
                                  lrate=.0001,
                                  lambda_l2=None,
                                  p_dropout=None):
    
    # Create base model
    model = create_cnn_network(image_size, nchannels,
                              name_base,
                              conv_layers,
                              conv_activation,
                              dense_layers,
                              dense_activation,
                              lrate,
                              lambda_l2,
                              p_dropout)
    # Output layer
    model.add(Dense(units=n_classes,
                    activation='softmax',
                    use_bias=True,
                    bias_initializer='zeros',
                    name='%soutput'%(name_base)))
    
    opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=.9, beta_2=0.999,
                                  epsilon=None, decay=0.0, amsgrad=False)
    
    # Asssuming right now two outputs
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                 metrics=['categorical_accuracy']) #, tf.keras.metrics.TruePositives(),
                         #tf.keras.metrics.FalsePositives(),
                         #tf.keras.metrics.TrueNegatives(),
                         #tf.keras.metrics.FalseNegatives()])
    return model

def training_set_generator_images(ins, outs, batch_size=10,
                          input_name='input', 
                        output_name='output'):
    '''
    Generator for producing random minibatches of image training samples.
    
    @param ins Full set of training set inputs (examples x row x col x chan)
    @param outs Corresponding set of sample (examples x nclasses)
    @param batch_size Number of samples for each minibatch
    @param input_name Name of the model layer that is used for the input of the model
    @param output_name Name of the model layer that is used for the output of the model
    '''
    
    while True:
        # Randomly select a set of example indices
        example_indices = random.choices(range(ins.shape[0]), k=batch_size)
        
        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield({input_name: ins[example_indices,:,:,:]},
             {output_name: outs[example_indices,:]})
        