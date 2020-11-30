# Author Qing Niu Feb 13th


import tensorflow as tf
import pandas as pd
import argparse
import numpy as np
import pickle 
import timeit
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time

# Tensorflow 2.0 way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

# Code from the class

#import symbiotic_metrics
#import hw2_base
# from deep_networks import *



# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

# build model with keras includes three hidden layers and [8,8,7] neurals, default activation function is elu, loss function is mse...

def shallow_network(n_inputs, hidden, n_output, activation='elu',dropout=None, dropout_input=None, lrate=0.001,kernel_regularizer = None,metrics = None):
	model = Sequential();
	model.add(InputLayer(input_shape=(n_inputs,)))
	for ind_hidden in range(len(hidden)):
		model.add(Dense(hidden[ind_hidden],use_bias=True, name="hidden"+str(ind_hidden), activation = activation))
	model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
	opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='mse', optimizer=opt,metrics= metrics)
	print(model.summary())

	return model
   # model.add(Dense(n_hidden, use_bias=True, name="hidden_1", activation=activation))
   # model.add(Dense(n_hidden, use_bias=True, name="hidden_2", activation=activation))
#     model.add(Dense(n_hidden-1, use_bias=True, name="hidden_3", activation=activation))
#     model.add(Dense(n_hidden, use_bias=True, name="hidden_4", activation=activation))
#     model.add(Dense(n_hidden, use_bias=True, name="hidden_5", activation=activation))
#     model.add(Dense(n_hidden, use_bias=True, name="hidden_6", activation=activation))
    
