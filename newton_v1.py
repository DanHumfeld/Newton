#################################################################
# Code      Newton's First Law via PINN with FI
# Version   1.0
# Date      2022-07-14
# Author    Dan Humfeld, DanHumfeld@Yahoo.com
# Note      This code solves F(t) = m*a(t) for x(t)
#           Easy, right?
#
#################################################################
# Importing Libraries
#################################################################

# # Enable this section to hide all warnings and errors
# import os
# import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
# import absl.logging

import math
import random
import numpy as np
from numpy import savetxt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from time import time
start_time = time()

#################################################################
# Inputs 
#################################################################   
# File names
output_model_X = 'model_X_part.h5'
output_model_F = 'model_F_air.h5'
prediction_results = 'model_X_t.csv'
loss_history = 'model_loss_history.csv'

# Optimizer for training: SGD, RSMProp, Adagrad, Adam, Adamax...
learning_rate = 0.001 #0.0005
my_optimizer = optimizers.Adam(learning_rate)
initializer = 'glorot_uniform'

# Mode: Train new = 0, continue training = 1, predict = 2
train_mode = 0

# Epoch and batch size control
epochs = 40000
batch = 1000

# Model hyper-parameters
nodes_per_layer = 16

# Normalization methods: 
# 0: none
# 3: scale by loss ratio
normalization_method = 0
loss_terms = 5
loss_ratio_limit = 1e-1

# Reporting options
plot_loss = False
time_reporting = True

#################################################################
# Problem Statement
#################################################################
# A block with mass 1 is at rest on a horizontal, frictionless 
# surface. Determine the force, as a function of time, that should
# be applied to move the block to specific locations at specific
# times. The jerk (dF/dt) may not exceed 50 N/s.

#################################################################
# Properties
#################################################################
mass = 1               # kg

#################################################################
# Range establishment
#################################################################
t_min = 0               # s
t_max = 6               # s
J_max = 50              # m/s^3

#################################################################
# Constraints (hard coded for now)
#################################################################
# constraint_X is tuples of [t,x,weighting]
constraint_X = [[0,  0, 1],
                [1,  4, 1],
                [4,  5, 1]]

# constraint_F is tuples of [t,F,weighting]
constraint_F = [[0, 0, 1]]

#################################################################
# Flatten method
#################################################################
def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

#################################################################
# Loss Normalization Methods
#################################################################
def determine_normalizations(losses, current_norms):
    # losses are not normalized coming in to this function
    ## Caclulate each loss/norm. Determine max/min ratio. If it's too big, renorm. If it's too small and norms aren't 1, renorm.
    ## When renorming, if the maximum normalization factor is small enough, set them all to 1; you've converged
    ## When renorming, you have to preserve the order of the losses, so use a power function - actually, no you don't, so don't get clever

    # For each loss determine if it needs a norm, instead of using the max to determine if anyone needs one
    # If it needs a norm, the equation is set up incorrectly; it gives 1e-4 at minimum for the normalization; you want it to result in loss/lossmax being 1e-4

    max_loss = max(losses)
    new_norms = current_norms
    for i in range(0,len(new_norms)):
        loss_ratio = losses[i]/max_loss
        if (loss_ratio > loss_ratio_limit):
            new_norms[i] = 1
        if (loss_ratio < loss_ratio_limit):
            new_norms[i] = (loss_ratio)/(loss_ratio_limit)
    return new_norms

#################################################################
# Building or Load Models
#################################################################
if (train_mode == 0):
    input1 = keras.layers.Input(shape=(1,))         # t

    layer1 = input1
    layer2 = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer1)   
    layer3 = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer2)
    output = keras.layers.Dense(1, activation = 'linear', kernel_initializer=initializer, bias_initializer=initializer)(layer3)           # Block position

    model_X = keras.models.Model([input1], [output])
    model_X.compile(loss='mse', optimizer=my_optimizer)


    input1 = keras.layers.Input(shape=(1,))         # t
    layer1a = input1
    layer2a = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer1a)   
    layer3a = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer2a)
    outputa = keras.layers.Dense(1, activation = 'linear', kernel_initializer=initializer, bias_initializer=initializer)(layer3a)           # Force

    model_F = keras.models.Model([input1], [outputa])
    model_F.compile(loss='mse', optimizer=my_optimizer)
else:
    model_X = keras.models.load_model(output_model_X) 
    model_F = keras.models.load_model(output_model_F) 

#################################################################
# Main Code
#################################################################
if (train_mode < 2):
    #Create Graph
    xdata = []
    ydata = []
    timedata = []
    if plot_loss:
        thatplot = plt.figure()
        thatplot.show()
        thatplot.patch.set_facecolor((0.1,0.1,0.1))
        axes = plt.gca()
        axes.set_xlim(0, 10)
        axes.set_ylim(0, +1)
        axes.set_facecolor((0.1,0.1,0.1))
        axes.spines['bottom'].set_color((0.9,0.9,0.9))
        axes.spines['top'].set_color((0.9,0.9,0.9))
        axes.spines['left'].set_color((0.9,0.9,0.9))
        axes.spines['right'].set_color((0.9,0.9,0.9))
        axes.xaxis.label.set_color((0.9,0.9,0.9))
        axes.yaxis.label.set_color((0.9,0.9,0.9))
        axes.tick_params(axis='x', colors=(0.9,0.9,0.9))
        axes.tick_params(axis='y', colors=(0.9,0.9,0.9))
        line, = axes.plot(xdata, ydata, 'r-') 

    min_loss = 100
    last_time = time()
    loss_weightings = np.zeros(loss_terms)

    for i in range(0, epochs):
        # Create tensors to feed to TF
        t_arr = np.random.uniform(0, t_max, batch) # Not normalized in this code... should it be?

        t_feed = np.column_stack((t_arr)) 
        t_feed = tf.Variable(t_feed.reshape(len(t_feed[0]),1), trainable=True, dtype=tf.float32)

        zero_feed = np.column_stack(np.zeros(len(t_arr)))
        zero_feed = tf.Variable(zero_feed.reshape(len(zero_feed[0]),1), trainable=True, dtype=tf.float32)

        constraint_X_feed = np.array(constraint_X)
        constraint_X_target = np.column_stack(constraint_X_feed[:,1])
        constraint_X_target = tf.Variable(constraint_X_target.reshape(len(constraint_X_target[0]),1), trainable=True, dtype=tf.float32)
        constraint_F_feed = np.array(constraint_F)
        constraint_F_target = np.column_stack(constraint_F_feed[:,1])
        constraint_F_target = tf.Variable(constraint_F_target.reshape(len(constraint_F_target[0]),1), trainable=True, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape_3:    
            with tf.GradientTape(persistent=True) as tape_2:  
                with tf.GradientTape(persistent=True) as tape_1:
                    # Watch parameters
                    tape_1.watch(t_feed)
                    tape_1.watch(zero_feed)
                    # Define functions
                    outputs_X = model_X([t_feed])
                    X_equ = outputs_X
                    outputs_F = model_F([t_feed])
                    F_equ = outputs_F
                    X_BC0 = model_X([zero_feed])

                # Watch parameters
                tape_2.watch(t_feed)
                # Take derivitives
                dX_dt = tape_1.gradient(X_equ, [t_feed])[0]
                dF_dt = tape_1.gradient(F_equ, [t_feed])[0]
                dX_dt0 = tape_1.gradient(X_BC0, [zero_feed])[0]

            # Take derivitives
            d2X_dt2 = tape_2.gradient(dX_dt, [t_feed])[0]  
            # Model losses
            # PDE
            loss_PDE_list = k.square(F_equ - mass * d2X_dt2)
            # Boundary conditions
            loss_BC0_list = k.square(dX_dt0 - 0)
            # Specified constraints         
            loss_X_constraint_list = tf.multiply(k.square(model_X([constraint_X_feed[:,0]]) - constraint_X_target), np.array(constraint_X_feed[:,2]).reshape(len(constraint_X_feed),1))
            loss_F_constraint_list = tf.multiply(k.square(model_F([constraint_F_feed[:,0]]) - constraint_F_target), np.array(constraint_F_feed[:,2]).reshape(len(constraint_F_feed),1))
            # Jerk constraint
            loss_Jerk_list = k.square(k.relu(k.abs(dF_dt) - 40)) 

            loss_PDE = k.mean(loss_PDE_list)
            loss_BC0 = k.mean(loss_BC0_list)
            loss_X_constraint = k.mean(loss_X_constraint_list)*len(loss_PDE_list)/len(loss_X_constraint_list)
            loss_F_constraint = k.mean(loss_F_constraint_list)*len(loss_PDE_list)/len(loss_F_constraint_list)
            loss_Jerk = k.mean(loss_Jerk_list)
            #loss_Jerk = 0 # Turning off the jerk loss for now
            losses = [loss_PDE, loss_BC0, loss_X_constraint, loss_F_constraint, loss_Jerk]
            #print([loss.numpy() for loss in losses])

            if (normalization_method==3):
                loss_weightings = determine_normalizations(losses, loss_weightings)
            if (normalization_method==0):
                loss_weightings = np.ones(len(losses))
            loss_total = sum([x*y for (x,y) in zip(losses, loss_weightings)])
            #print(format(loss_total.numpy(), ".3f"))

        # Train the model
        #print([format(loss.numpy(), ".2e") for loss in losses], [format(weight, ".2f") for weight in loss_weightings])
        gradients_X = tape_3.gradient(loss_total, model_X.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients_X, model_X.trainable_variables) if grad is not None) 
        gradients_F = tape_3.gradient(loss_total, model_F.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients_F, model_F.trainable_variables) if grad is not None) 

        # Take a break and report
        i_loss = sum(losses)
        if i % 100 == 0:
            print("Step " + str(i) + " -------------------------------")
            '''
            print("Loss_PDE:  ", "{:.3e}".format(k.get_value(losses[0])))
            print("Loss_X_t:  ", "{:.3e}".format(k.get_value(losses[1])))
            print("Loss_F_t:  ", "{:.3e}".format(k.get_value(losses[2])))
            print("Loss_Jerk: ", "{:.3e}".format(k.get_value(losses[3])))
            '''
            print(["{:.3e}".format(k.get_value(loss)) for loss in losses])
            #print(["{:.3e}".format(loss_w) for loss_w in loss_weightings])
            print("Loss_tot: ", "{:.3e}".format(i_loss))
            if (time_reporting):
                print("Calculation time for last period: ", "{:.0f}".format(round(time() - last_time, 0)))
            last_time = time()
            
            #Report
            i_time = round(time() - start_time,2) 
            xdata.append(i)
            ydata.append(i_loss)
            timedata.append(i_time)
            if plot_loss:
                line.set_xdata(xdata)
                line.set_ydata(ydata)
                plt.draw()
                plt.pause(1e-17)
                plt.xlim(0,i)
                plt.ylim(i_loss/5,i_loss*10)  

            savetxt(loss_history, np.column_stack((timedata,xdata,ydata)), comments="", header="Time(s),Epoch,Total Loss", delimiter=',', fmt='%f')
            #Only save model if loss is improved
            if ((min_loss > i_loss) or (normalization_method==4)):
                min_loss = i_loss
                model_X.save(output_model_X)
                model_F.save(output_model_F)

    model_X.save(output_model_X)
    model_F.save(output_model_F)

#################################################################
# Predicting
#################################################################
# Entire part needs to be rewritten (sometime) (probably)

# Inputs
nodes = 101
t_feed = np.arange(nodes)*(t_max/nodes)
report_X = model_X.predict([t_feed])
report_F = model_F.predict([t_feed])
results = np.column_stack((t_feed,report_X,report_F))
np.savetxt(prediction_results, results, delimiter=',') 

print("Job's done")
