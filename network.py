#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:40:47 2020
5043_hw4
@author: qingn
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt

from itertools import product
import hw2_base

from tensorflow import keras
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import RMSprop
import random
import re

# From pypng
import png
#from sklearn.p
import sklearn.metrics

##################
# Configure figure parameters

FONTSIZE = 18
FIGURE_SIZE = (10,4)
FIGURE_SIZE2 = (10,10)

plt.rcParams.update({'font.size': FONTSIZE})
plt.rcParams['figure.figsize'] = FIGURE_SIZE
# Default tick label size
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


# Image Reading
def readPngFile(filename):
    '''
    Read a single PNG file
    
    filename = fully qualified file name
    
    Return: 3D numpy array (rows x cols x chans)
    
    Note: all pixel values are floats in the range 0.0 .. 1.0
    
    This implementation relies on the pypng package
    '''
    #print("reading:", filename)
    # Load in the image meta-data
    r = png.Reader(filename)
    it = r.read()
    
    # Load in the image itself and convert to a 2D array
    image_2d = np.vstack(map(np.uint8, it[2]))
    
    # Reshape into rows x cols x chans
    image_3d = np.reshape(image_2d,
                         (it[0],it[1],it[3]['planes'])) / 255.0
    return image_3d

def read_images_from_directory(directory, file_regexp):
    '''
    Read a set of images from a directory.  All of the images must be the same size
    
    directory = Directory to search
    
    file_regexp = a regular expression to match the file names against
    
    Return: 4D numpy array (images x rows x cols x chans)
    '''
    
    print(directory, file_regexp)
    # Get all of the file names
    files = sorted(os.listdir(directory))
    
    # Construct a list of images from those that match the regexp
    list_of_images = [readPngFile(directory + "/" + f) for f in files if re.search(file_regexp, f) ]
    
    # Create a 3D numpy array
    return np.array(list_of_images, dtype=np.float32)

def read_image_set_from_directories(directory, spec):
    '''
    Read a set of images from a set of directories
    
    directory  = base directory to read from
    
    spec = n x 2 array of subdirs and file regexps
    
    Return: 4D numpy array (images x rows x cols x chans)
    
    '''
    out = read_images_from_directory(directory + "/" + spec[0][0], spec[0][1])
    for sp in spec[1:]:
        out = np.append(out, read_images_from_directory(directory + "/" + sp[0], sp[1]), axis=0)
    return out

def load_multiple_image_sets_from_directories(directory_base, directory_list, object_list, test_files):
    '''
    
    '''
    print("##################")
    # Create the list of object/image specs
    inputs = [[obj, test_files] for obj in object_list]
    
    # First directory
    ret = read_image_set_from_directories(directory_base + "/" + directory_list[0], inputs)
    
    # Loop over directories
    for directory in directory_list[1:]:
        ret = np.append(ret,
                        read_image_set_from_directories(directory_base + "/" + directory, inputs),
                        axis=0)

    return ret
### Create Network
def create_classifier_network(image_size, nchannels, n_classes=2, lambda_l2=.0001, p_dropout=0.5):
    '''
    This is to create a deep network with the argParser
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size[0],image_size[1], nchannels),name ='input'))
    
   
    ### Fill in detail
    for j in range(4):
        model.add(Convolution2D(filters = args.conv_nfilters[j], 
                            kernel_size = (args.conv_size[j],args.conv_size[j]), 
                            strides =(1,1), 
                            padding='valid', 
                            use_bias = True, 
                            kernel_initializer='random_uniform',
                            bias_initializer = 'zeros',
                            name = 'C%s'%(j), 
                            activation = 'elu',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))# valid is cutting down the size
        model.add(MaxPooling2D(pool_size= (2,2),
                           strides=(2,2),
                           padding = 'valid'                     
                           ))
    model.add(Flatten())
    for i in range(3):
        model.add(Dense(units = args.hidden[i],activation='elu',use_bias=True,kernel_initializer='truncated_normal',
                   bias_initializer='zeros',name='D_%s'%(i),kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
        if p_dropout is not None:
            model.add(Dropout(p_dropout))
#    model.add(Dropout(p_dropout))
        i = i+1
        
    model.add(Dense(units=n_classes,
                   activation='softmax',
                    use_bias=True,
                    bias_initializer='zeros',
                    kernel_initializer='truncated_normal',name = 'output',
                   kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)
             ))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
#     keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss = 'categorical_crossentropy',optimizer =opt, metrics=['accuracy'])
    print(model.summary())
    
    return model

def shallow_create_classifier_network(image_size, nchannels, n_classes=2, lambda_l2=.0001, p_dropout=0.5):
    '''
    This is to create a shallow network with the argParser
    '''
    
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size[0],image_size[1], nchannels),name ='input'))
    
   
    ### Fill in detail
    for j in range(4):
        model.add(Convolution2D(filters = args.conv_nfilters[j], 
                            kernel_size = (args.conv_size[j],args.conv_size[j]), 
                            strides =(1,1), 
                            padding='valid', 
                            use_bias = True, 
                            kernel_initializer='random_uniform',
                            bias_initializer = 'zeros',
                            name = 'C%s'%(j), 
                            activation = 'elu',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))# valid is cutting down the size
        model.add(MaxPooling2D(pool_size= (2,2),
                           strides=(2,2),
                           padding = 'valid'                     
                           ))
    model.add(Flatten())
    for i in range(3):
        model.add(Dense(units = args.hidden[i],activation='elu',use_bias=True,kernel_initializer='truncated_normal',
                   bias_initializer='zeros',name='D_%s'%(i),kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
        if p_dropout is not None:
            model.add(Dropout(p_dropout))
#    model.add(Dropout(p_dropout))
        i = i+1
        
    model.add(Dense(units=n_classes,
                   activation='softmax',
                    use_bias=True,
                    bias_initializer='zeros',
                    kernel_initializer='truncated_normal',name = 'output',
                   kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)
             ))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
#     keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss = 'categorical_crossentropy',optimizer =opt, metrics=['accuracy'])
    print(model.summary())
    
    return model


def training_set_generator_images(ins, outs, batch_size=10,
                          input_name='input', 
                        output_name='output'):
    '''
    Generator for producing random mini-batches of image training samples.
    
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
        yield({input_name: ins[example_indices,:,:,:]},{output_name: outs[example_indices,:]})
### Evaluation
def generate_roc(model, ins, outs, ins_validation, outs_validation):
    '''
    Produce a ROC plot given a model, a set of inputs and the true outputs
    
    Assume that model produces N-class output; we will only look at the class 0 scores
    '''
    # Compute probabilistic predictions given images
    pred = model.predict(ins)
    # Compute false positive rate & true positive rate + AUC
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(outs[:,0], pred[:,0])
    auc = sklearn.metrics.auc(fpr, tpr)
    
    # Compute probabilistic predictions given images
    pred_val = model.predict(ins_validation)
    # Compute false positive rate & true positive rate + AUC
    fpr_val, tpr_val, thresholds_val = sklearn.metrics.roc_curve(outs_validation[:,0], pred_val[:,0])
    auc_val = sklearn.metrics.auc(fpr_val, tpr_val)
    
    
    # Generate the plot
    plt.figure(1)
    plt.axis('equal')
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr, 'b', label='Train AUC = {:.3f}'.format(auc))
    plt.plot(fpr_val, tpr_val, 'r', label='Validation AUC = {:.3f}'.format(auc_val))
    plt.legend(loc='best')
    plt.xlabel('FPR', fontsize=FONTSIZE)
    plt.ylabel('TPR', fontsize=FONTSIZE)
    plt.savefig(os.getcwd()+'ccn_spectra_avg'+'2017-11-1*'+str(i)+'.png') 
            
# generate_roc(model, ins, outs, ins_validation, outs_validation)  

### Visualize Model Internals      
def intermediate_model_state(model, ins, layer_list):
    '''
    Return layer activations for intermediate layers in a model for a set of examples
    
    :param model: Model in question
    :param ins: Input tensor (examples, rows, cols, channels)
    :param layer_list: List of layer names to produce activations for
    :returns: a list of numpy arrays
    '''
    # Translate layer names into corresponding output tensors
    layer_outputs = [l.output for l in model.layers if l.name in layer_list]
    
    # Construct a new Keras model that outputs these tensors
    # The internal structure of the model itself is referenced through the input and output tensor lists
    new_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Evaluate the new model
    activations = new_model.predict(ins_validation)
    
    # Return a list of activation numpy arrays
    return activations

def visualize_state(activations, width=1, example=0, cmap='plasma'):
    '''
    Produce graphical representation of a set of image channels
    
    :param activations: numpy array (example, rows, cols, channels)
    :param width: Number of images displayed horizontally
    :param example: Index of example to display
    :param cmap: Color map to use for plotting
    '''
    # Size of the individual images
    nrows = activations.shape[1]
    ncols = activations.shape[2]
    # Number of channels
    nfilters = activations.shape[3]
    
    # Tile all of the sub-images 
    grid = np.zeros((int((nfilters-1)/width + 1) * nrows, ncols * width))
    
    # Loop over image
    for i in range(nfilters):
        # Compute r,c of tile to place the ith image into
        r = int(i / width)
        c = i % width
        grid[nrows*r: nrows*(r+1), ncols*c:ncols*(c+1)] = activations[example,:,:,i]
        
    # Plot
    plt.matshow(grid, cmap=cmap) 
    
# Compute activations for 2 layers over a set of examples
#layer_list=['C1']
#activations = intermediate_model_state(model, ins_validation, layer_list)
## Plot convolutional layers 1 and 2
#example=29
#plt.imshow(ins_validation[example,:,:,:])
#visualize_state(activations, width=10, example=example)
##visualize_state(activations[0], width=10, example=example)
##visualize_state(activations[1], width=20, example=example)
##visualize_state(activations[2], width=20, example=example)
##visualize_state(activations[3], width=30, example=example)

#%%
if __name__ == "__main__":
    q_parser = hw2_base.create_parser()

    args = q_parser.parse_args()
    hw2_base.check_args(args)
#    hw2_base.execute_exp(args)

#%%
'''Load data sets'''
    
## File location
# directory_base = '/home/fagg/datasets/core50/core50_128x128'
directory_base = '/Users/qingn/Downloads/core50_128x128'

# Training set: define which files to load for each object
#test_files = '.*[05].png'
test_files = '.*0.png'

### Positive cases
# Define which objects to load
#object_list = ['o25', 'o22', 'o23', 'o24']
object_list = ['o15', 'o12', 'o13', 'o14']
#object_list = ['o13']

# Define which conditions to load
condition_list = ['s1', 's2', 's3', 's4', 's5', 's7', 's8', 's9', 's10', 's11']
#condition_list = ['s1', 's2', 's3', 's4']
#condition_list = ['s1', 's2','s3', 's4']

# Load all of the objects/condition
ins_pos = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list, test_files)

### Negative cases
# Define which objects to load
object_list2 = ['o45', 'o42', 'o43', 'o44']
#object_list2 = ['o42']
ins_neg = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list2, test_files)

### Combine positives and negatives into a common data set
outs_pos = np.append(np.ones((ins_pos.shape[0],1)), np.zeros((ins_pos.shape[0],1)), axis=1)
outs_neg = np.append(np.zeros((ins_neg.shape[0],1)), np.ones((ins_neg.shape[0],1)), axis=1)

ins = np.append(ins_pos, ins_neg, axis=0)
outs = np.append(outs_pos, outs_neg, axis=0)
#%%
########################################################################
# Validation set
# Define which files to load for each object
test_files = '.*0.png'

### Positives
# Define which objects to |load
object_list = ['o11']
#object_list = ['o21']

# Load the positives
ins_pos_validation = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list, test_files)

### Negatives
# Define objects
object_list2 = ['o41']
#object_list2 = ['o41']

# Load the negative cases
ins_neg_validation = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list2, test_files)

### Combine positives and negatives
outs_pos_validation = np.append(np.ones((ins_pos_validation.shape[0], 1)), np.zeros((ins_pos_validation.shape[0], 1)), axis=1)
outs_neg_validation = np.append(np.zeros((ins_neg_validation.shape[0], 1)), np.ones((ins_neg_validation.shape[0], 1)), axis=1)

ins_validation = np.append(ins_pos_validation, ins_neg_validation, axis=0)
outs_validation = np.append(outs_pos_validation, outs_neg_validation, axis=0)
#%%


history=[0]*5
model=[0]*5
generator = [0]*5
#plt.figure()
for i in range(2):

    generator[i] = training_set_generator_images(ins, outs, batch_size=args.batch)#
    
    model[i] = create_classifier_network((ins.shape[1], ins.shape[2]), ins.shape[3], 2)
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                         min_delta=0.01)
    
    # Learn
    history[i] = model[i].fit_generator(generator[i],
                                  epochs=int(args.epochs*3),
                                  steps_per_epoch=2,
                                  use_multiprocessing=True, 
                                  verbose=args.verbose>=2,
                                  validation_data=(ins_validation, outs_validation),
                                  callbacks=[early_stopping_cb])
#    history_all.append(history)
    

#%
#run network.py -vv -epochs 2000 -patience 400 -conv_size 3 5 5 5 -conv_nfilters 10 15 20 25 -hidden 200 50 10 -L2_regularizer .003 -lrate 0.0002 -batch 200
    plt.plot(history[i].history['val_accuracy'])
#    plt.show()
    
    print(history[i].history['val_accuracy'])
    #%%
for i in range(2):
    plt.figure()
    generate_roc(model[i], ins, outs, ins_validation, outs_validation)
    #%%
#a = []    
#qas=[0]*4
#model=[0]*4
#for i in range(4):
#    qas[i] = training_set_generator_images(ins, outs, batch_size=args.batch)#
#    model[i] = create_classifier_network((ins.shape[1], ins.shape[2]), ins.shape[3], 2)
#    a.append(i)
    
 #%%

            
# generate_roc(model, ins, outs, ins_validation, outs_validation)  
#run network.py -vv -epochs 200 -patience 200 -conv_size 4 5 5 5 -conv_nfilters 10 15 20 25 -hidden 200 100 50 10 -L2_regularizer 0.003 -lrate 0.0002 -batch 200 -dropout 0.6
#def simpleGeneratorFun(): 
#
#    i = 
#        yield {'wo':i},{'shi':i*9}
#        
#ss= simpleGeneratorFun()
#
#for i in ss:
#    print(i)
## Driver code to check above generator function 
##%%
#for i in range(5):
#    
#    for value in generator:  
#        print(value) 
#    i=i+1