#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:54:05 2025

@author: johnserences
@author: hkular
"""

# imports
import numpy as np
import os
import json
import warnings
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from helper_funcs import *
from rdk_task import RDKtask
import math
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
from decode_funcs import decode_rnn

# set up some nice colors (from Robert and Nuttida)
hex_c = ['#06D2AC', '#206975', '#6F3AA4', '#2B1644']

#--------------------------
# Basic model params
#--------------------------
n_models = np.arange(1)           # number of models with these parameters...make iterable so can make non-contiguous list if you want
device = 'cpu'                  # device to use for loading/eval model
task_type = 'rdk'                 # task type (conceptually think of as a motion discrimination task...)         
n_afc = 3                         # number of stimulus alternatives
T = 200                           # timesteps in each trial
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_prob_model = 0.5             # for model training: probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
stim_prob_task = 1/n_afc          # for task eval         
stim_amps = 1.0
stim_noise = 0.5                   # can make this a list of amps and loop over... 
batch_size = 1000                 # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct
out_size = 1

# init dict of task related params  - init here because it contains train_amp
settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_task, 'stim_amp' : stim_amps, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size}
# create the task object
task = RDKtask( settings )


#--------------------------
# Define SVM and grid 
#--------------------------
n_cvs = 3 # cross-val folds
n_cgs = 15 # penalties to eval
Cs = np.logspace( -5,1,n_cgs )
param_grid = { 'C': Cs, 'kernel': ['linear'] } # set up grid
grid = GridSearchCV( SVC(class_weight='balanced'), param_grid, refit=True, verbose=0, n_jobs = math.floor(os.cpu_count() * 0.7) )
window = 50 # for sliding average over time
#n_boots = 1000 # number of times to decode for boot_strap resampling
# dict of decoding params    
decode_settings = {'n_cvs':n_cvs, 'n_cgs': n_cgs, 'Cs': Cs, 'param_grid':param_grid, 'grid':grid, 'window':window}
# Initialize decoding settings
decode = decode_rnn( settings, decode_settings )



for m_idx,m_num in enumerate( n_models ):
    
    # which type of model are we decoding?
    model_name = f'num_afc-{n_afc}_stim_prob-{int( stim_prob_model * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-10_nw_mse_modnum-{m_num}'
    # what file are we saving?
    npz_name = f'decoding/{model_name}.npz'
    
    # check if decoding .npz exists, if so continue to next m_idx
    if os.path.exists(npz_name):# the npz file I would be saving exists already 
        warnings.warn(f"File '{npz_name}' already exists. Skipping model {m_num}.")
        continue
    else:
    
        # build a file name...
        fn = f'trained_models/{model_name}.pt'
    
        # load the trained model, set to eval, requires_grad == False
        net = load_model( fn,device )
        
        # eval a batch of trials using the trained model
        outputs,s_label,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3,m_acc,tbt_acc = eval_model( net, task )
        print( f'Eval batch done for model {m_num}' )
        
        h_map = { 1: h1, 2: h2, 3: h3 } # h1 is [time,batch_size,unit]
        
        all_acc = []
        # Iterate through layers
        for l_idx, data in h_map.items():
                   
            # split trials into train and test
            tri_ind, hold_out = decode.split_train_test(data)
           
            # do decoding
            acc = decode.decode_times( data, s_label, tri_ind, hold_out )
            print( f'Done decoding for layer {l_idx} for model {m_num}' )
            all_acc.append( acc )
            
        all_acc = np.stack( all_acc ) # (n_layers, n_times, n_stim)
            
        # save output for each model       
        np.savez( npz_name, all_acc = all_acc, tbt_acc = tbt_acc )

    
# In[]
# check jsons and flag if decoding accuracy is not at criteria
directory = os.getcwd()    
if task_type == 'rdk':
    fulldir = f'{directory}/trained_models_rdk'
elif task_type == 'rdk_reproduction':
    fulldir = f'{directory}/trained_models_rdk_reproduction'
    
for filename in os.listdir(fulldir):
    if filename.endswith('.json'):
        # look at rnn_settings from json
        with open(f'{os.path.join(fulldir, filename)}', 'r') as infile:
            settings, rnn_settings = json.load(infile)                    
        if rnn_settings['running_acc'] >= rnn_settings['acc_crit']:
            print(f"running acc: {rnn_settings['running_acc']}")
            continue
        else:
            print(f"Too low! running acc: {rnn_settings['running_acc']} for model: {model_name}")
       
# In[]
# compare my trained_models directory with the neurocube

def compare_directories(dir1, dir2):
    # Get sets of filenames in each directory (excluding subdirectories)
    files_dir1 = set(f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f)))
    files_dir2 = set(f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f)))

    # Find missing files
    missing_in_dir2 = files_dir1 - files_dir2
    missing_in_dir1 = files_dir2 - files_dir1

    # Report
    if missing_in_dir2:
        print(f"{len(missing_in_dir2)} file(s) missing in '{dir2}':")
        for f in sorted(missing_in_dir2):
            print(f"  {f}")
    else:
        print(f"No files missing in '{dir2}'.")

    if missing_in_dir1:
        print(f"{len(missing_in_dir1)} file(s) missing in '{dir1}':")
        for f in sorted(missing_in_dir1):
            print(f"  {f}")
    else:
        print(f"No files missing in '{dir1}'.")

# Example usage
compare_directories('/Users/hkular/Documents/GitHub/expectation_rnn/trained_models/reproduction task', '/Volumes/serenceslab/john/expectation_rnn_2.0/trained_models')
   
# In[]

#plot inputs and outputs and targets

# after initialize task with specifics
u, s_label = task.generate_rdk_stim()

idx = s_label[1]  # This gives the correct channel index (0, 1, or 2) for trial 0
trial = 1         # We're plotting the first trial
for ch in range(u.shape[2]):
    if ch == idx:
        plt.plot(u[:, trial, ch], label=f'Stim {idx}')
    else:
        plt.plot(u[:, trial, ch])

plt.xlabel('Time Steps')
plt.ylabel('Input (au)')
plt.legend()
plt.show()
    
# In[]
# plot average over models plus individual models

plt.figure(10,9)
for m_num in enumerate ( n_models ):
    
    npz_name = f'decoding/num_afc-{n_afc}_stim_prob-{int( stim_prob_model * 100 )}_stim_amp-{int( stim_amps * 100 )}_nw_mse_modnum-{m_num}.npz'
    acc = np.load(npz_name) # n_layers, n_times, n_stim
    all_acc.append(acc)
np.stack(all_acc)

# which layer are we plotting?
layer = 1
   
# mean accuracy over models
y_expected = np.mean( acc[:,layer,:,0], axis = 0 )
y_unexpected = np.mean( np.mean( acc[:,layer,:,1:], axis = -1 ), axis=0 )
        

# Define the confidence interval (95%) over models
CI_expected = st.t.interval( alpha=.95, df=acc.shape[0]-1, loc=y_expected, scale=st.sem(acc[:,layer,:,0], axis=0))
CI_unexpected = st.t.interval( alpha=.95, df=acc.shape[0]-1, loc=y_unexpected, scale=st.sem(np.mean(acc[:,layer,:,1:], axis = -1), axis=0))

# Define the x-axis times
window_indices = decode.sliding_window()
x_time = [np.rint(np.mean(np.arange(0,T,1)[inds])) for inds in window_indices] #np.arange(y_expected.shape[1])

# Plotting
#plt.figure(figsize=(10, 6))

# Plot the mean line
plt.plot(x_time, y_expected, label='Expected', color='blue')
plt.plot(x_time, y_unexpected, label='Unexpected', color='red')

# Plot the confidence interval as a ribbon
plt.fill_between(x_time, y_expected - CI_expected, y_expected + CI_expected, color='blue', alpha=0.3, label='95% CI')
plt.fill_between(x_time, y_unexpected - CI_unexpected, y_unexpected + CI_unexpected, color='red', alpha=0.3, label='95% CI')

# Plot faint individual model lines
for m_num in enumerate (n_models):
    plt.plot(x_time, acc[m_num, layer, :, 0], color = 'blue', alpha = 0.2)
    plt.plot(x_time, np.mean( acc[m_num,layer,:,1:], axis = -1 ), color = 'red', alpha = 0.2)

# Labels and title
plt.xlabel('Time')
plt.ylabel('Decoding accuracy (%)')
plt.title(f'{self.n_afc} afc {self.stim_amp} coh')
plt.legend()



# plot
#decode.plot_decode_models(acc, layer=1, save=FALSE)
    
# In[]
######### GARBAGE In GARBAGE Out test 

#import random


#random.shuffle( s_label )
#acc_g = decode.decode_times( data, s_label, tri_ind, hold_out )
#decode.plot_decoding_single(acc_g)
   
       
# read an npz file and take a look  
#new_acc = np.load(npz_name) 
    
    
    
    
    
    