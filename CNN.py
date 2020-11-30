#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:34:51 2020

@author: qingn
"""

    #%%
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
#%%

directory = './htdma_figures_new'
# directory = '/home/fagg/datasets/core50/core50_128x128/s1/o21'
files = os.listdir(directory)
#%%
r = png.Reader(directory + "/" + files[0])
it = r.read()
image_2d = np.vstack(map(np.uint8, it[2]))
image_3d = np.reshape(image_2d,
                         (60,4,4))

plt.imshow(image_3d)
#%%
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
#%%
    
class JobIterator():
    
    
    def __init__(self,params):
        '''
        Constructor
        
        @param params Dictionary of key/list pairs
        '''
        self.params = params
        self.product = list(dict(zip(params,x))for x in product(*params.values()))
        self.iter = (dict(zip(params,x))for x in product(*params.values()))
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        @return The next combination in the list
        '''
        return self.next()
    def get_index(self,i):
        '''
        Return the ith combination of parameters
        
        @param i Index into the Cartesian product list
        @return The ith combination of parameters
        '''
        return self.product[i]
    def get_njobs(self):
        '''
        @return The total number of combinationss
        '''
        return len(self.product)
    def set_attributes_by_index(self,i,obj):
        '''
        For an arbitrary object, set the attributes to match the ith job parameters
        
        @param i Index into the Cartesian product list
        @param obj Arbitrary object (to be modified)
        '''# For an arbitrary object, set the attributes to match the ith job parameters
        
        d = self.get_index(i)
        
        for k,v in d.items():
        
            setattr(obj,k,v)
 #%%
directory_base = './htdma_figures_new'

# Training set: define which files to load for each object
#test_files = '.*[05].png'
test_files = '.*[0].png'
#%%
object_list = ['o13']
condition_list =['']
ins_pos = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list, test_files)

#%%
### Negative cases
# Define which objects to load
#object_list2 = ['o45', 'o42', 'o43', 'o44']
object_list2 = ['o12']
ins_neg = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list2, test_files)

### Combine positives and negatives into a common data set
outs_pos = np.append(np.ones((ins_pos.shape[0],1)), np.zeros((ins_pos.shape[0],1)), axis=1)
outs_neg = np.append(np.zeros((ins_neg.shape[0],1)), np.ones((ins_neg.shape[0],1)), axis=1)

ins = np.append(ins_pos, ins_neg, axis=0)
outs = np.append(outs_pos, outs_neg, axis=0)
#%%
# Validation set
# Define which files to load for each object
test_files = '.*.png'

### Positives
# Define which objects to load
object_list = ['o31']
#object_list = ['o21']

# Load the positives
ins_pos_validation = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list, test_files)

### Negatives
# Define objects
object_list2 = ['o21']
#object_list2 = ['o41']

# Load the negative cases
ins_neg_validation = load_multiple_image_sets_from_directories(directory_base, condition_list, object_list2, test_files)

### Combine positives and negatives
outs_pos_validation = np.append(np.ones((ins_pos_validation.shape[0], 1)), np.zeros((ins_pos_validation.shape[0], 1)), axis=1)
outs_neg_validation = np.append(np.zeros((ins_neg_validation.shape[0], 1)), np.ones((ins_neg_validation.shape[0], 1)), axis=1)

ins_validation = np.append(ins_pos_validation, ins_neg_validation, axis=0)
outs_validation = np.append(outs_pos_validation, outs_neg_validation, axis=0)
#%%
print(ins.shape)
print(outs.shape)
print(ins_validation.shape)
print(outs_validation.shape)
#%% Create network
def create_classifier_network(image_size, nchannels, n_classes=2, lambda_l2=.0001, p_dropout=0.5):
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels)))
    
    model.add(Convolution2D(filters = 12, 
                            kernel_size = (2,1), 
                            strides =(1,1), 
                            padding='valid', 
                            use_bias = True, 
                            kernel_initializer='random_uniform',
                            bias_initializer = 'zeros',
                            name = 'C1', 
                            activation = 'elu',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))# valid is cutting down the size
    model.add(MaxPooling2D(pool_size= (2,1),
                           strides=(1,1),
                           padding = 'valid'                     
                           ))        
    model.add(Convolution2D(filters = 10, 
                            kernel_size = (2,1), 
                            strides =(1,1), 
                            padding='valid', 
                            use_bias = True, 
                            kernel_initializer='random_uniform',
                            bias_initializer = 'zeros',
                            name = 'C2', 
                            activation = 'elu',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))# valid is cutting down the size
    model.add(MaxPooling2D(pool_size= (1,1),
                           strides=(1,1),
                           padding = 'valid'                     
                           ))

    model.add(Flatten())
    model.add(Dense(units = 100,activation='elu',use_bias=True,kernel_initializer='truncated_normal',
                   bias_initializer='zeros',name='D1',kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))

    model.add(Dropout(p_dropout))
    model.add(Dense(units=n_classes,
                   activation='softmax',
                    use_bias=True,
                    kernel_initializer='truncated_normal',
                   kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)            ))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)#     keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss = 'binary_crossentropy',optimizer =opt, metrics=['accuracy'])
    print(model.summary())
    
    return model
#%%
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
        yield({input_name: ins[example_indices,:,:,:]},
             {output_name: outs[example_indices,:]})
#%%
        
if __name__ == "__main__":
    parser = hw2_base.create_parser()
    args = parser.parse_args()
    hw2_base.check_args(args)
#    execute_exp(args)

#%%
        # Training generator
generator = training_set_generator_images(ins, outs, batch_size=args.batch)#
#%%
#%%
model = create_classifier_network((ins.shape[1], ins.shape[2]), ins.shape[3], 2)


# Learn
history = model.fit_generator(generator,
                              epochs=args.epochs,
                              steps_per_epoch=2,
                              use_multiprocessing=True, 
                              verbose=args.verbose>=2,
                              validation_data=(ins_validation, outs_validation) )
                              #,callbacks=[early_stopping_cb])
#%%
                              
model = create_classifier_network((ins.shape[1], ins.shape[2]), ins.shape[3], 2)
#%%
history = model.fit(x=ins, y=outs, epochs=700, verbose=1,use_multiprocessing=True,
                        validation_data=(ins_validation, outs_validation))#, 
#                        callbacks=[early_stopping_cb])
#%%
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
    #%%
generate_roc(model, ins, outs, ins_validation, outs_validation)

#%%
plt.plot(history.history['accuracy'],label = 'accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('CNN accuracy')
plt.ylabel('accuracy')
plt.legend()
plt.xlabel('epoch')