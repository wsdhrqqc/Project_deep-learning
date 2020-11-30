import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time

import argparse
import pickle

# Tensorflow 2.0 way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################
def build_model(n_inputs, n_hidden, n_output, activation='elu', lrate=0.001):
    model = Sequential();
    model.add(InputLayer(input_shape=(n_inputs,)))
    model.add(Dense(n_hidden, use_bias=True, name="hidden", activation=activation))
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
    
    opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=opt)
    print(model.summary())
    return model

########################################################
def execute_exp(exp_index, epochs):

    ##############################
    # Run the experiment
    # Create training set: XOR
    ins = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outs = np.array([[0], [1], [1], [0]])
    
    model = build_model(ins.shape[1], 2, outs.shape[1], activation='sigmoid')

    # Callbacks
    #checkpoint_cb = keras.callbacks.ModelCheckpoint("xor_model.h5",
    #                                                save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=100,
                                                 restore_best_weights=True,
                                                 min_delta=.00001)

    # Training
    history = model.fit(x=ins, y=outs, epochs=epochs, verbose=False,
                        validation_data=(ins, outs),
                        callbacks=[early_stopping_cb])

    # Report
    mse = history.history['loss']

    fp = open("xor_results_%02d.pkl"%(exp_index), "wb")
    pickle.dump(mse, fp)
    fp.close()

    # Display
    #plt.plot(history.history['loss'])
    #plt.ylabel('MSE')
    #plt.xlabel('epochs')
    #plt.savefig('xor_perf.png')


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='XOR Learner')
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    args = parser.parse_args()

    execute_exp(args.exp, args.epochs)
