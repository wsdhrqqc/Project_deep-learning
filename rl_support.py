'''
Support for RL experiments

Author: Andrew H. Fagg
'''
import sys
sys.path.append("../../../tf_tools/networks")
from cnn_classifier import *


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import tensorflow.keras as keras

#from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, UpSampling1D, Input, InputLayer, Reshape, Activation, Lambda, AveragePooling1D, Subtract
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import random
#import skimage.transform as sktr
import gym
from mpl_toolkits.mplot3d import Axes3D
import re
 

#from sklearn.p
import sklearn.metrics

from sklearn.utils.extmath import cartesian

####################################

FONTSIZE = 18
FIGURE_SIZE = (10,4)
FIGURE_SIZE2 = (10,10)

# Configure parameters
plt.rcParams.update({'font.size': FONTSIZE, 'figure.figsize': FIGURE_SIZE})

# Default tick label size
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE


class numpyBuffer:
    '''
    Circular buffer using a numpy array
    
    In this case, we only append to this buffer and overwrite values once we wrap-around
    '''
    def __init__(self, maxsize=100, ndims=1, dtype=np.float32):
        '''
        Constructor for the buffer
        
        :param maxsize: Maximum number of rows that can be stored in the buffer
        :param ndims: The number of columns in the buffer       
        '''
        
        self.buffer = np.zeros((maxsize,ndims), dtype=dtype)
        self.maxsize=maxsize
        self.ndims=ndims
        self.back = 0
        self.full = False
    
    def size(self):
        '''
        :return: The number of items stored in the buffer
        '''
        if(self.full):
            return self.maxsize
        else:
            return self.back
        
    def append(self, rowvec):
        '''
        Append a row to the buffer
        
        :param rowvec: Numpy row vector of values to append.  Must be 1xndims
        '''
        self.buffer[self.back,:] = rowvec
        self.back = self.back+1
        if self.back >= self.maxsize:
            self.back = 0
            self.full = True
            
    def getrows(self, row_indices):
        '''
        Return a set of indicated rows
        
        :param row_indices: Array of row indices into the buffer
        :return: len(row_indices)xndims numpy array
        '''
        return self.buffer[row_indices,:]
    
class numpyTensorBuffer:
    '''
    Circular buffer using a numpy array
    
    In this case, we only append to this buffer and overwrite values once we wrap-around
    '''
    def __init__(self, maxsize=100, dims=None, dtype=np.float32):
        '''
        Constructor for the buffer
        
        :param maxsize: Maximum number of rows that can be stored in the buffer
        :param dims: A tuple that is the shape of the individual instances     
        '''
        
        self.buffer = np.zeros((maxsize,) + dims, dtype=dtype)
        self.maxsize=maxsize
        self.dims=dims
        self.back = 0
        self.full = False
    
    def size(self):
        '''
        :return: The number of items stored in the buffer
        '''
        if(self.full):
            return self.maxsize
        else:
            return self.back
        
    def append(self, tensor):
        '''
        Append a tensor to the buffer
        
        :param tensor: Numpy tensor of values to append.  Must be 1xdims
        '''
        self.buffer[self.back,:] = tensor
        self.back = self.back+1
        if self.back >= self.maxsize:
            self.back = 0
            self.full = True
            
    def getobjects(self, inds):
        '''
        Return a set of indicated rows
        
        :param row_indices: Array of row indices into the buffer
        :return: len(row_indices)xdims numpy array
        '''
        return self.buffer[inds,:]
    
class myAgent:
    def __init__(self, state_size, action_size, action_continuous=None, epsilon=.01, gamma=0.99, 
                 lrate=.001, maxlen=10000, use_done=True, greedy_divisor=10):
        '''
        :param state_size: Number of state variables
        :param action_size: Number of actions (will use one-hot encoded actions)
        :param action_continuous: List of continuous actions that correspond to the discrete choices.  If None, then
                we have a built-in set of discrete actions
        :param epsilon: Constant exploration rate
        :param gamma: Constant discout rate
        :param lrate: Learning rate
        :param action_discrete: Network produces one Q-value for each discrete action 
                (True is the only supported case)
        :param maxlen: Maximum length of the circular experience buffer
        :param use_done: Use the done signal to handle the TD error differently in done/not cases
        
        Experience buffer is designed for quick access to prior experience
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.action_continuous = action_continuous
        self.epsilon=epsilon
        self.gamma=gamma
        self.reward_log = []
        self.greedy_reward_log = []
        self.verbose = False
        self.verbose_execute = False
        self.lrate=lrate
        self.action_discrete=(action_continuous == None)
        self.log_observation = numpyBuffer(maxlen, state_size)
        self.log_observation_new = numpyBuffer(maxlen, state_size)
        self.log_action = numpyBuffer(maxlen, 1, dtype=np.int16)
        self.log_reward = numpyBuffer(maxlen, 1)
        self.log_done = numpyBuffer(maxlen, 1, dtype=np.bool)
        self.use_done = use_done
        self.greedy_divisor = greedy_divisor
        
    def build_model_predictor(self, n_units, activation='elu', lambda_regularization=None):
        '''
        Simple sequential model.
        
        :param n_units: Number of units in each hidden layer (a list)
        :param activation: Activation function for the hidden units
        :param lambda_regularization: None or a continuous value (currently not used)
        '''
        model = Sequential()
        i = 0
        
        # Regularization
        if lambda_regularization is not None:
            lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)
            
        # Input layer
        model.add(InputLayer(input_shape=(self.state_size,)))
        
        # Loop over hidden layers
        for n in n_units:
            model.add(Dense(n, 
                        activation=activation,
                        use_bias=True,
                        kernel_initializer='truncated_normal', 
                        bias_initializer='zeros', 
                        name = "D"+str(i),
                        kernel_regularizer=lambda_regularization))
            i=i+1
            
        # model.add(BatchNormalization())
        # Output layer
        model.add(Dense(self.action_size, 
                        activation=None,
                        use_bias=True,
                        kernel_initializer='truncated_normal', 
                        bias_initializer='zeros',  
                        name = "output",
                        kernel_regularizer=lambda_regularization))
        
        return model
        
    def build_model_parallel(self, n_units, activation='elu', 
                             lambda_regularization=None):
        '''
        Simple sequential model.
        
        :param n_units: Number of units in each hidden layer (a list)
        :param gamma: gamma parameter value
        :param activation: Activation function for the hidden units
        :param lambda_regularization: None or a continuous value (currently not used)
        '''
        self.model_input_observation = Input(shape=(self.state_size,), name='observation')
        self.model_input_observation_next = Input(shape=(self.state_size,), name='observation_next')
        self.model_input_action = Input(shape=(1,), dtype=np.int32, name='action')
        self.model_input_done = Input(shape=(1,), dtype=np.bool, name='done')
        self.model_gamma = keras.backend.constant(self.gamma, name='gamma')
                                                  
        # Create an instance of the Q-function
        model = self.build_model_predictor(n_units=n_units, activation=activation, 
                                           lambda_regularization=lambda_regularization)
        self.model = model
        
        # Q = r + gamma * Q_next_max
        # Reorganize:
        #    Q - gamma * Q_next_max = r (not done)  
        # OR Q = r (done)
        
        # Estimate the Q values for the current and next states
        model_q = model(self.model_input_observation)
        model_q_next = model(self.model_input_observation_next)
        
        # Take the q-values for the selected actions (one for each example)
        # Each row is represented exactly once
        rows = tf.range(tf.shape(model_q)[0])
        rows = tf.reshape(rows, shape=(-1,1))

        # Selected actions are the columns
        # Construct list of coordinates in the matrix
        inds = tf.concat([rows, self.model_input_action], axis=1)
        
        # Select the q-values for the executed actions
        model_q_selected = tf.gather_nd(model_q, inds, name='q_selected')
        
        ######
        # Compute max value for each next state: example x 1
        model_q_next_max = tf.keras.backend.max(model_q_next, axis=1)
        model_q_next_max = tf.reshape(model_q_next_max, shape=(-1,1))

        # Compute 1-done
        select = 1.0 - tf.keras.backend.cast(self.model_input_done, dtype=np.float32)
        
        # gamma * select * Q_next
        model_q_next_modified = self.gamma * model_q_next_max * select
        
        # Estimate r = Q_selected - gamma * select * Q_next
        self.model_output_r = keras.layers.subtract([model_q_selected,  model_q_next_modified], name='output_r')
        
        # We have a separate model for learning
        self.model_learning = Model(inputs=[self.model_input_observation, self.model_input_observation_next,
                                           self.model_input_action, self.model_input_done], \
                                    outputs=self.model_output_r)
        
        # Configure model
        opt = keras.optimizers.Adam(lr=self.lrate, beta_1=0.9, beta_2=0.999, 
                            epsilon=None, decay=0.0, amsgrad=False)
        
        self.model_learning.compile(loss='mse', optimizer=opt)
        
        # Indicate that we have the parallel architecture
        self.simple = False
        
        print(model.summary())
        print(self.model_learning.summary())
        
    def build_model(self, n_units, activation='elu',
                    lambda_regularization=None):
        '''
        Simple sequential model.
        
        :param n_units: Number of units in each hidden layer (a list)
        :param activation: Activation function for the hidden units
        :param lambda_regularization: None or a continuous value (currently not used)
        '''
        model = self.build_model_predictor(n_units=n_units, activation=activation, lambda_regularization=lambda_regularization)
        self.model = model
        
        # We do not have a separate model for learning
        self.model_learning = None
        
        # Configure model
        opt = keras.optimizers.Adam(lr=self.lrate, beta_1=0.9, beta_2=0.999, 
                            epsilon=None, decay=0.0, amsgrad=False)
        
        model.compile(loss='mse', optimizer=opt)
        
        self.simple = True
        
        print(model.summary())
        
        
    def choose_action(self, observation, verbose=False, greedy_flag=False):
        '''
        epsilon-greedy choice of discrete action
        
        :returns: (discrete_action, explore_bit)

        '''
        if(not greedy_flag and np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size), True
        else:
            pred = self.model.predict(observation)[0]
            if verbose:
                print(pred)
            return np.argmax(pred), False
    
    def choose_action_continuous(self, observation, verbose=False, greedy_flag=False):
        '''
        epsilon-greedy choice of continuous action
        
        :returns: (discrete_action, continuous_action, explore_bit)
        '''
        action_index, explore = self.choose_action(observation, verbose, greedy_flag=greedy_flag)
        return action_index, self.action_continuous[action_index], explore
    
    def log_experience(self, observation, action_index, reward, observation_new, done):
        ''' 
        Store the last step in the circular buffer
        '''
        observation =  np.array(observation, ndmin=2)
        observation_new =  np.array(observation_new, ndmin=2)
        
        self.log_observation.append(observation)
        self.log_observation_new.append(observation_new)
        self.log_action.append(action_index)
        self.log_reward.append(reward)
        self.log_done.append(done)
                
    def learning_step(self, batch_size=200):
        '''
        Iterate over a minibatch of the stored experience & take a learning step with each

        :param batch_size: Size of the batch to do learning with
        
        '''
        
        # Sample from the prior experience.  How we do this depends on how much
        #  experience that we have accumulated so far
        if self.log_observation.size() < batch_size:
            minibatch_inds = range(self.log_observation.size())
            #return
        else:
            # Random sample from the buffer
            minibatch_inds = random.sample(range(self.log_observation.size()), batch_size)
        
        print("Creating batch:", len(minibatch_inds))
        
        # Extract the logged values
        observations = self.log_observation.getrows(minibatch_inds)
        observations_new = self.log_observation_new.getrows(minibatch_inds)
        rewards = self.log_reward.getrows(minibatch_inds)[:,0]
        dones = self.log_done.getrows(minibatch_inds)[:,0]  
        actions = self.log_action.getrows(minibatch_inds)[:,0]
        
        if self.simple:
            # Standard Q implementation
            # Update targets: for each example, only one action is updated
            #  (the one that was actually executed)
            q = targets = self.model.predict(observations)
            
            q_next = self.model.predict(observations_new)
            q_next_max = np.max(q_next, axis=1)
        
            if self.use_done:
                done_list = np.argwhere(dones)
                done_not_list = np.argwhere(np.logical_not(dones))
        
                # Last step in the episodes
                targets[done_list, actions[done_list]] = rewards[done_list]
                # Other steps
                targets[done_not_list, actions[done_not_list]] = rewards[done_not_list] \
                    + self.gamma * q_next_max[done_not_list]
            else:
                targets[:, actions] = self.gamma * q_next_max
        
            # Update the Q-function
            self.model.fit(observations, targets, epochs=1, verbose=0)
        else:
            # Parallel model
            self.model_learning.fit({'observation': observations, 
                                    'observation_next': observations_new,
                                    'action': actions,
                                    'done': dones},
                                   rewards,
                                   epochs=1,
                                   verbose=0)

        if self.verbose:
            print(observations, targets)
           
    
    def execute_trial(self, env, nsteps, render_flag=False, batch_size=100, greedy_flag=False):
        '''
        A trial terminates at nsteps or when the environment says we must stop.
        
        '''
        observation = env.reset()
        observation = np.array(observation, ndmin=2)
        # Accumulator for total reward
        reward_total = 0
        
        # Loop over each step
        for i in range(nsteps):
            if render_flag:
                env.render()
                
            # Some environments require discrete actions, while others require continous actions
            if self.action_discrete:
                action_index, explore = self.choose_action(observation, verbose=self.verbose_execute,
                                                          greedy_flag=greedy_flag)
                observation_new, reward, done, info = env.step(action_index) 
            else:
                # Figure out which action to execute
                action_index, action_continuous, explore = self.choose_action_continuous(observation, verbose=self.verbose_execute,greedy_flag=greedy_flag)
                observation_new, reward, done, info = env.step(action_continuous)
                
            # Remember reward
            reward_total = reward_total + reward
            if self.verbose_execute:
                print(observation, action_index, reward, observation_new, done)
                
            # Log this step 
            #if not explore:
            if not greedy_flag:
                self.log_experience(observation, action_index, reward, 
                                        observation_new, done)
                
            if done:
                # Environment says we are done
                break
                
            # Prepare for the next step
            observation = observation_new
            observation = np.array(observation, ndmin=2)
            
        # Learning
        #print("before learning")
        if not greedy_flag:
            self.learning_step(batch_size=batch_size)
            # Log accumulated reward for this trial
            self.reward_log.append(reward_total)
            
            # Every greedy_divisor learning step, do a greedy step, too
            if(len(self.reward_log)%self.greedy_divisor == 0):
               self.execute_trial(env, nsteps, render_flag=render_flag, batch_size=batch_size, greedy_flag=True)
        else:
            # This is a greedy step: no learning & a separate log
            self.greedy_reward_log.append((len(self.reward_log), reward_total))
            print('Greedy')   
        if render_flag:
            env.close()
        #print(reward_total)
        
        
        
    def execute_ntrials(self, env, ntrials, nsteps, render_flag=False, batch_size=100, greedy_flag=False):
        '''
        Execute the specified number of trials
        '''
        for _ in range(ntrials):
            self.execute_trial(env, nsteps, render_flag, batch_size, greedy_flag=greedy_flag)
        

###############################################################
class myImageAgent:
    def __init__(self, state_shape, action_size, action_continuous=None, 
                 epsilon=.01, gamma=0.99, 
                 lrate=.001, maxlen=10000, 
                 greedy_divisor=10, 
                 target_model_divisor=None,
                action_repeat=4, 
                blend_images=True):
        '''
        :param state_shape: Shape of the input image
        :param action_size: Number of actions (will use one-hot encoded actions)
        :param action_continuous: List of continuous actions that correspond to the discrete choices.  If None, then
                we have a built-in set of discrete actions
        :param epsilon: Constant exploration rate
        :param gamma: Constant discout rate
        :param lrate: Learning rate
        :param action_discrete: Network produces one Q-value for each discrete action 
                (True is the only supported case)
        :param maxlen: Maximum length of the circular experience buffer
        :param greedy_divisor: Number of learning trials between greedy (non-learning trials)
        :param target_model_divisor: Number of training trials that we hold the target model constant
        :param policy_gradient: Use the policy gradient algorith (default: Q-learning)
        :param action_repeat: The number of times that a single action is repeated in the environment
        
        Experience buffer is designed for quick access to prior experience
        '''
        self.state_shape = state_shape
        self.action_size = action_size
        self.action_continuous = action_continuous
        self.epsilon=epsilon
        self.gamma=gamma
        self.reward_log = []
        self.greedy_reward_log = []
        self.verbose = False
        self.verbose_execute = False
        self.lrate=lrate
        self.action_discrete=(action_continuous == None)
        self.log_observation = numpyTensorBuffer(maxlen, state_shape, dtype=np.uint8)
        self.log_observation_new = numpyTensorBuffer(maxlen, state_shape, dtype=np.uint8)
        self.log_action = numpyBuffer(maxlen, 1, dtype=np.int16)
        self.log_reward = numpyBuffer(maxlen, 1)
        self.log_done = numpyBuffer(maxlen, 1, dtype=np.bool)
        self.greedy_divisor = greedy_divisor
        self.trial = 0
        self.target_model_divisor = target_model_divisor
        self.action_repeat = action_repeat
        self.blend_images = blend_images
            
    def build_model(self, 
                    conv_layers=[],
                    conv_activation='elu',
                    dense_layers=[],
                    dense_activation='elu',
                    lrate=0.0001,
                    lambda_l2=None,
                    p_dropout=None):
        '''        
        :param n_units: Number of units in each hidden layer (a list)
        :param activation: Activation function for the hidden units
        :param lambda_regularization: None or a continuous value (currently not used)
        '''
   
        model = create_cnn_network(self.state_shape[0:2], self.state_shape[2], 
                                  'CNN',
                                  conv_layers,
                                  conv_activation,
                                  dense_layers,
                                  dense_activation,
                                  lrate,
                                  lambda_l2,
                                  p_dropout)
        
        
        # Q-learning
        # Output layer
        model.add(Dense(self.action_size, 
                        activation=None,
                        use_bias=True,
                        kernel_initializer='truncated_normal', 
                        bias_initializer='zeros',  
                        name = "output"))
                        #kernel_regularizer=keras.regularizers.l2(lambda_regularization),
                        #bias_regularizer=keras.regularizers.l2(lambda_regularization)))
        # Configure model
        opt = keras.optimizers.Adam(lr=self.lrate, beta_1=0.9, beta_2=0.999, 
                            epsilon=None, decay=0.0, amsgrad=False)
        
        model.compile(loss='mse', optimizer=opt)
        
        self.model = model
        
        print(model.summary())
        if self.target_model_divisor is None:
            # The model and the target model are the same
            self.target_model = self.model
        else:
            # The target model is separate than the learned model
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
    
            
    def choose_action(self, observation, verbose=False, greedy_flag=False):
        '''
        epsilon-greedy choice of discrete action
        
        :returns: (discrete_action, explore_bit)

        '''
        if(greedy_flag or np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size), True
        else:
            #pred = self.model.predict(observation)[0]
            pred = self.model.predict(np.reshape(observation, (1,)+observation.shape))[0]

            if verbose:
                print(pred)
            return np.argmax(pred), False
    
    def choose_action_continuous(self, observation, verbose=False, greedy_flag=False):
        '''
        epsilon-greedy choice of continuous action
        
        :returns: (discrete_action, continuous_action, explore_bit)
        '''
        action_index, explore = self.choose_action(observation, verbose, greedy_flag=greedy_flag)
        return action_index, self.action_continuous[action_index], explore
    
    def log_experience(self, observation, action_index, reward, observation_new, done):
        ''' 
        Store the last step in the circular buffer
        '''
        observation =  np.array(observation, ndmin=2)
        observation_new =  np.array(observation_new, ndmin=2)
        
        self.log_observation.append(observation)
        self.log_observation_new.append(observation_new)
        self.log_action.append(action_index)
        self.log_reward.append(reward)
        self.log_done.append(done)
                
    def learning_step(self, batch_size=200):
        '''
        Iterate over a minibatch of the stored experience & take a learning step with each

        :param batch_size: Size of the batch to do learning with
        
        '''
        # Count number of trials
        self.trial = self.trial+1
        
        # Sample from the prior experience.  How we do this depends on how much
        #  experience that we have accumulated so far
        if self.log_observation.size() < batch_size:
            minibatch_inds = range(self.log_observation.size())
            #return
        else:
            # Random sample from the buffer
            minibatch_inds = random.sample(range(self.log_observation.size()), batch_size)
        
        print("Creating batch:", len(minibatch_inds))
        
        # Extract the logged values
        observations = self.log_observation.getobjects(minibatch_inds)
        observations_new = self.log_observation_new.getobjects(minibatch_inds)
        rewards = self.log_reward.getrows(minibatch_inds)[:,0]
        dones = self.log_done.getrows(minibatch_inds)[:,0]  
        actions = self.log_action.getrows(minibatch_inds)[:,0]
        
        #if self.simple:
        # Standard Q implementation
        # Update targets: for each example, only one action is updated
        #  (the one that was actually executed)
        q = targets = self.model.predict(observations)
            
        q_next = self.target_model.predict(observations_new)
        q_next_max = np.max(q_next, axis=1)
        
        done_list = np.argwhere(dones)
        done_not_list = np.argwhere(np.logical_not(dones))
        
        # Last step in the episodes
        targets[done_list, actions[done_list]] = rewards[done_list]
        # Other steps
        targets[done_not_list, actions[done_not_list]] = rewards[done_not_list] \
                + self.gamma * q_next_max[done_not_list]
            
        # Update the Q-function
        self.model.fit(observations, targets, epochs=1, verbose=0)
            
        # Update the target model?
        if((self.target_model_divisor is not None) and (self.trial % self.target_model_divisor)==0):
            #Yes
            self.target_model.set_weights(self.model.get_weights())
            print("Update target")
                
        #else:
            ### NO LONGER USING
            # Parallel model
            #self.model_learning.fit({'observation': observations, 
            #                        'observation_next': observations_new,
            #                        'action': actions,
            #                        'done': dones},
            #                       rewards,
            #                       epochs=1,
            #                       verbose=0)

        if self.verbose:
            print(observations, targets)
           
    def execute_trial(self, env, nsteps, render_flag=False, batch_size=100, greedy_flag=False):
        '''
        A trial terminates at nsteps or when the environment says we must stop.
        
        '''
        observation = env.reset()
        observation = np.array(observation)
        # Accumulator for total reward
        reward_total = 0
        
        # Loop over each step
        for _ in range(nsteps):
            if render_flag:
                env.render()
                

            action_index, explore = self.choose_action(observation, verbose=self.verbose_execute,
                                                          greedy_flag=greedy_flag)
            observation_new, reward, done, info = env.step(action_index) 
            
            #TODO: check this...
            j = 1
            while not done and j < self.action_repeat:
                j = j + 1
                # Execute the same action a second time & then combine the results
                observation_new2, reward2, done2, info2 = env.step(action_index) 
            
                reward=reward+reward2
                done = done2
                if self.blend_images:
                    observation_new = observation_new / 2 + observation_new2 / 2
                else:
                    observation_new = observation_new2
                
            # Remember reward
            reward_total = reward_total + reward
            if self.verbose_execute:
                print(observation, action_index, reward, observation_new, done)
                
            # Log this step 
            #if not explore:
            if not greedy_flag:
                self.log_experience(observation, action_index, reward, 
                                        observation_new, done)
                
            if done:
                # Environment says we are done
                break
                
            # Prepare for the next step
            observation = observation_new
            observation = np.array(observation)
            
        # Report result
        print(reward_total)
        # Learning
        #print("before learning")
        if not greedy_flag:
            self.learning_step(batch_size=batch_size)
            # Log accumulated reward for this trial
            self.reward_log.append(reward_total)
            
            # Every greedy_divisor learning step, do a greedy step, too
            if(len(self.reward_log)%self.greedy_divisor == 0):
               self.execute_trial(env, nsteps, render_flag=render_flag, batch_size=batch_size, greedy_flag=True)
        else:
            # This is a greedy step: no learning & a separate log
            self.greedy_reward_log.append((len(self.reward_log), reward_total))
            print('Greedy')   
        print("done learning")
        
        if render_flag:
            env.close()
        
        
        
        
    def execute_ntrials(self, env, ntrials, nsteps, render_flag=False, batch_size=100):
        '''
        Execute the specified number of trials
        '''
        for _ in range(ntrials):
            self.execute_trial(env, nsteps, render_flag, batch_size)
            
    def save_model(self, name):
        self.model.save(name)
        
    def plot_curves(self):
        '''Plot the learning curves'''
        ### Show accumulated reward as a function of trial
        # Raw data
        plt.plot(self.reward_log)

        # Greedy policy
        x = [e[0] for e in self.greedy_reward_log]
        y = [e[1] for e in self.greedy_reward_log]
        plt.plot(x,y, 'r')

        # Average e-greedy results
        l = int(len(self.reward_log)/self.greedy_divisor)
        avgs = np.mean(np.reshape(self.reward_log[:l*self.greedy_divisor], newshape=(l,-1)), axis=1)
        x = (np.arange(avgs.shape[0])+1)*self.greedy_divisor

        plt.plot(x,avgs, 'k')
       
            
###########################################
# Policy Gradient
def policy_gradient_loss(y_true, y_pred):
    '''
    :param y_true: accumulated reward
    :param y_pred: PI of selected action
    '''
    # Log of PI.  Constant ensures that log() behaves
    log = keras.backend.log(y_pred + .001)
    
    # mean of log*R.  Negative because we are minimizing the loss
    L = -keras.backend.mean(log * y_true)
    return L
    
class myImagePolicyGradientAgent:
    def __init__(self, state_shape, action_size, 
                 epsilon=.01, gamma=0.99, 
                 lrate=.001, maxlen=10000, 
                 greedy_divisor=10,
                action_repeat=4,
                blend_images=True):
        '''
        :param state_shape: Shape of the input image
        :param action_size: Number of actions (will use one-hot encoded actions)
        :param action_continuous: List of continuous actions that correspond to the discrete choices.  If None, then
                we have a built-in set of discrete actions
        :param epsilon: Constant exploration rate
        :param gamma: Constant discout rate
        :param lrate: Learning rate
        :param action_discrete: Network produces one Q-value for each discrete action 
                (True is the only supported case)
        :param maxlen: Maximum length of the circular experience buffer
        :param greedy_divisor: Number of learning trials between greedy (non-learning trials)
        :param target_model_divisor: Number of training trials that we hold the target model constant
        :param action_repeat: Number of times a single action is repeatedly executed in the environment
        
        Experience buffer is designed for quick access to prior experience
        '''
        self.state_shape = state_shape
        self.action_size = action_size
        self.epsilon=epsilon
        self.gamma=gamma
        self.reward_log = []
        self.greedy_reward_log = []
        self.verbose = False
        self.verbose_execute = False
        self.lrate=lrate
        self.log_observation = numpyTensorBuffer(maxlen, state_shape, dtype=np.uint8)
        self.log_action = numpyBuffer(maxlen, 1, dtype=np.int16)
        self.log_reward = numpyBuffer(maxlen, 1)
        self.log_done = numpyBuffer(maxlen, 1, dtype=np.bool)
        self.greedy_divisor = greedy_divisor
        self.trial = 0
        self.action_repeat = action_repeat
        self.blend_images = blend_images
   
    
    def build_model(self, 
                    conv_layers=[],
                    conv_activation='elu',
                    dense_layers=[],
                    dense_activation='elu',
                    lrate=0.0001,
                    lambda_l2=None,
                    p_dropout=None):
        
        # Use standard engine to create the image processing + dense component
        model = create_cnn_network(self.state_shape[0:2], self.state_shape[2], 
                                  'CNN',
                                  conv_layers,
                                  conv_activation,
                                  dense_layers,
                                  dense_activation,
                                  lrate,
                                  lambda_l2,
                                  p_dropout)
       
        # Policy gradient
        # Output layer
        model.add(Dense(self.action_size, 
                    activation='softmax',
                    use_bias=True,
                    kernel_initializer='truncated_normal', 
                    bias_initializer='zeros',  
                    name = "output"))

        # Learning Model
        # Input: image + selected action
        model_input = Input(shape=self.state_shape, name='image')
        model_action = Input(shape=(1,), name='selected_action', dtype=np.int32)
        
        # Pi values for all actions
        pi = model(model_input)
        
        # Select the PI values for the selected_action
        # Take the q-values for the selected actions (one for each example)
        # Each row is represented exactly once
        rows = tf.range(tf.shape(model_action)[0])
        rows = tf.reshape(rows, shape=(-1,1))
        inds = tf.concat([rows, model_action], axis=1)
        pi_selected = tf.gather_nd(pi, inds, name='pi_selected')
        pi_selected = tf.reshape(pi_selected, shape=(-1,1))
        
        # Create learning model
        model_learning = Model(inputs={'image': model_input, 'selected_action': model_action},
                              outputs=pi_selected)
        
        # Optimizer
        opt = keras.optimizers.Adam(lr=self.lrate, beta_1=0.9, beta_2=0.999, 
                         epsilon=None, decay=0.0, amsgrad=False)
        
        # Compile model
        model_learning.compile(loss=policy_gradient_loss, optimizer=opt)      
        
        self.model = model
        self.model_learning = model_learning
        
        # Report
        print(model.summary())
        print(model_learning.summary())
            
    def choose_action(self, observation, verbose=False, greedy_flag=False):
        '''
        epsilon-greedy choice of discrete action
        
        :returns: (discrete_action, explore_bit)

        '''
        if(not greedy_flag and np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size), True
        else:
            #pred = self.model.predict(observation)[0]
            pi = self.model.predict(observation)[0]

            if verbose:
                print(pi)
            return np.random.choice(self.action_size, p=pi), False
    
    def log_experience(self, observations, action_indices, rewards_accumulated, dones):
        ''' 
        Store the last step in the circular buffer
        '''
        #observation =  np.array(observation, ndmin=2)
        #observation_new =  np.array(observation_new, ndmin=2)
        
        for o in observations:
            self.log_observation.append(o)
        for a in action_indices:
            self.log_action.append(a)
        for r in rewards_accumulated:
            self.log_reward.append(r)
        for d in dones:
            self.log_done.append(d)
                
    def learning_step(self, batch_size=200):
        '''
        Extract a minibatch of the stored experience & take a learning step with each
        :param batch_size: Size of the batch to do learning with
        
        '''
        # Count number of trials
        self.trial = self.trial+1
        
        # Sample from the prior experience.  How we do this depends on how much
        #  experience that we have accumulated so far
        if (batch_size is None) or (self.log_observation.size() < batch_size):
            minibatch_inds = range(self.log_observation.size())
            #return
        else:
            # Random sample from the buffer
            minibatch_inds = random.sample(range(self.log_observation.size()), batch_size)
        
        print("Creating batch:", len(minibatch_inds))
        
        # Extract the logged values
        observations = self.log_observation.getobjects(minibatch_inds)
        rewards = self.log_reward.getrows(minibatch_inds)[:,0]
        dones = self.log_done.getrows(minibatch_inds)[:,0]  
        actions = self.log_action.getrows(minibatch_inds)[:,0]
        
        # Adjust rewards (only if there is interesting variance)
        std = np.std(rewards)
        if(std > .01):
            rewards = (rewards - np.mean(rewards))/std
        
        # Update the learning model
        self.model_learning.fit({'image': observations, 'selected_action': actions},
                          rewards, epochs=1, verbose=0)

        if self.verbose:
            print(observations, targets)
           
    def execute_trial(self, env, nsteps, render_flag=False, batch_size=100, greedy_flag=False):
        '''
        A trial terminates at nsteps or when the environment says we must stop.
        
        '''
        observation = env.reset()

        # Local logs (for this trial only)
        observations = []
        actions = []
        rewards = []
        dones = []
        
        # Loop over each step
        for i in range(nsteps):
            if render_flag:
                env.render()
                
            # Some environments require discrete actions, while others require continous actions
            action_index, explore = \
                self.choose_action(np.reshape(observation, newshape=(1,)+observation.shape), 
                                   verbose=self.verbose_execute,
                                    greedy_flag=greedy_flag)
            # Execute
            observation_new, reward, done, info = env.step(action_index) 

            # Repeat execution of the same action
            j = 1
            while not done and j < self.action_repeat:
                j = j + 1
                # Execute the same action a second time & then combine the results
                observation_new2, reward2, done2, info2 = env.step(action_index) 
            
                reward=reward+reward2
                done = done2
                
                if self.blend_images:
                    observation_new = observation_new / 2 + observation_new2 / 2
                else:
                    observation_new = observation_new2
            
            # Log this state (this is a local log!)
            observations.append(observation)
            rewards.append(reward)
            actions.append(action_index)
            dones.append(done)
            
            if done:
                # Environment says we are done
                break
                
            # Prepare for the next step
            observation = observation_new
        
        # Compute accumulated, discounted reward
        r_accum = 0
        reward_total = 0
        reward_accumulated = []
        
        # Interate over the rewards backwards
        for r in reversed(rewards):
            r_accum = r_accum * self.gamma + r
            reward_accumulated.append(r_accum)
            reward_total += r
        
        # Accumulated, discounted rewards (in correct order)
        reward_accumulated = np.array(reward_accumulated)[::-1]
                    
        # Report result
        print(reward_total)
        
        if not greedy_flag:
            # Learning
            self.log_experience(observations, actions, reward_accumulated, 
                                        dones)
            
            self.learning_step(batch_size=batch_size)
            # Log accumulated reward for this trial
            self.reward_log.append(reward_total)
            print("done learning")
            
            # It may be time to do a greedy trial...
            # Every greedy_divisor learning step, do a greedy step, too
            if((self.trial%self.greedy_divisor) == 0):
               self.execute_trial(env, nsteps, render_flag=render_flag, batch_size=batch_size, greedy_flag=True)
        else:
            # This is a greedy step: no learning & a separate log
            self.greedy_reward_log.append((len(self.reward_log), reward_total))
            print('Greedy')   
        
        if render_flag:
            env.close()
        
    def execute_ntrials(self, env, ntrials, nsteps, render_flag=False, batch_size=100):
        '''
        Execute the specified number of trials
        '''
        for _ in range(ntrials):
            self.execute_trial(env, nsteps, render_flag, batch_size)

    def save_model(self, name):
        self.model.save(name + "_query")
        self.model_learning.save(name + "_learning")
        
    def plot_curves(self):
        '''Plot the learning curves'''
        ### Show accumulated reward as a function of trial
        # Raw data
        plt.plot(self.reward_log)

        # Greedy policy
        x = [e[0] for e in self.greedy_reward_log]
        y = [e[1] for e in self.greedy_reward_log]
        plt.plot(x,y, 'r')

        # Average e-greedy results
        l = int(len(self.reward_log)/self.greedy_divisor)
        avgs = np.mean(np.reshape(self.reward_log[:l*self.greedy_divisor], newshape=(l,-1)), axis=1)
        x = (np.arange(avgs.shape[0])+1)*self.greedy_divisor

        plt.plot(x,avgs, 'k')
        