#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:00:15 2025

@author: hkular
"""
#--------------------------
# imports
#--------------------------
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from scipy.stats import sem
from helper_funcs import *
from rdk_task import RDKtask
import matplotlib.colors
from numpy import trapezoid
from model_count import count_models

# set up some nice colors (from Robert and Nuttida)
hex_c = ['#06D2AC', '#206975', '#6F3AA4', '#2B1644']

#--------------------------
# Basic model params
#--------------------------
device = 'cpu'                    # device to use for loading/eval model
task_type = 'rdk_repro_cue'                 # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                       # number of stimulus alternatives
T = 225                          # timesteps in each trial
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_prob = 1/n_afc          # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
stim_prob_to_eval = stim_prob          # eval the model at this prob level (stim_prob is used to determine which trained model to use)
stim_amps_train = 1.0                # can make this a list of amps and loop over... 
stim_amps = 1.0
stim_noise_train = 0.1
stim_noise = stim_noise_train                 # magnitude of randn background noise in the stim channel
batch_size = 1000                 # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct
weighted_loss = 0                       #  0 = nw_mse l2 or 1 = weighted mse
noise_internal = 0.1            # noise trained at 0.1
num_cues = 2
cue_on = 75 
cue_dur = T-cue_on
cue_layer = 1

if task_type == 'rdk':
    fn_stem = 'trained_models_rdk/gonogo_'
    out_size = 1
elif task_type == 'rdk_reproduction':
    fn_stem = 'trained_models_rdk_reproduction/repro_'
    out_size = n_afc  
elif task_type == 'rdk_repro_cue':
    fn_stem = f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer}/reprocue_'
    out_size = n_afc  



#--------------------------
# decoding params
#--------------------------
trn_prcnt = 0.8    # percent of data to use for training
n_cvs = 5          # how many train/test cv folds
classes = 'stim'   # which are we decoding stim or choice
time_or_xgen = 1   # decode timepnt x timepnt or do full xgen matrix 
w_size = 5         # mv_avg window size
num_cs = 1         # number of C's to grid search, if 1 then C=1
n_cvs_for_grid = 5 # num cv folds of training data to find best C
max_iter = 5000    # max iterations

#--------------------------
# init dict of task related params
# note that stim_prob_to_eval is passed in here
# and that the model name (fn) will be generated based 
# on stim_prob... 
#--------------------------
settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_to_eval, 'stim_amp' : stim_amps, 'stim_noise' : stim_noise, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':150}

# create the task object
task = RDKtask( settings )


#--------------------------
# How many trained models in this cond
#--------------------------
n_models = count_models(n_afc, stim_prob, stim_amps_train, stim_noise_train, weighted_loss, task_type, fn_stem, directory = os.getcwd())
n_layers = 3

#--------------------------
# matrices to store model decode acc
#--------------------------

if time_or_xgen == 0:
    over_acc = np.full( ( n_models,n_layers,T//w_size ),np.nan )
    stim_acc = np.full( ( n_models,n_layers,n_afc,T//w_size ),np.nan )
else:
    over_acc = np.full( ( n_models,n_layers,T//w_size,T//w_size ),np.nan )
    stim_acc = np.full( ( n_models,n_layers,n_afc,T//w_size,T//w_size ),np.nan ) 
      


for m_num in np.arange( n_models ):
    
    # build a file name...
    if weighted_loss == 0:
        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
    else:
        # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
        if stim_prob == 1 / n_afc:
            fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'
        else:
            fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'  

        
    # load the trained model, set to eval, requires_grad == False
    net = load_model( fn,device )
    # load cue scramble matrix
    if task_type == 'rdk_repro_cue':
        with open(f'{fn[:-3]}.json', "r") as infile:
           _ , rnn_settings = json.load(infile)
        sr_scram_list = rnn_settings['sr_scram']
        sr_scram_list = [sr_scram_list[str(s)] for s in sorted(sr_scram_list.keys(), key=int)]
        sr_scram = np.array(sr_scram_list)
    else:
        sr_scram = []
    print(f'loaded model {m_num}')
    
    # update internal noise
    net.recurrent_layer.noise = noise_internal

    # eval a batch of trials using the trained model
    outputs,s_label,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3,m_acc,tbt_acc,cues = eval_model( net, task, sr_scram )
    
    # s_label is a diff shape for cue version, deal with if statement later
    s_label_int = np.argmax(s_label, axis=1)
    
    layer_data = [h1, h2, h3]
    #  decoding
    for l_num in np.arange(n_layers):
        
        if time_or_xgen == 0:
            tmp_over_acc = np.full((n_cvs,T//w_size),np.nan)
            tmp_stim_acc = np.full((n_cvs,n_afc,T//w_size),np.nan)
        else:
            tmp_over_acc = np.full( ( n_cvs,T//w_size,T//w_size ),np.nan )
            tmp_stim_acc = np.full( ( n_cvs,n_afc,T//w_size,T//w_size ),np.nan )    
        
        
        for cv in range( n_cvs ):
            if classes == 'stim':
                tmp_over_acc[cv,:], tmp_stim_acc[cv,:,:] = decode_ls_svm(layer_data[l_num], s_label_int, n_afc, w_size, time_or_xgen, trn_prcnt)          
            elif classes == 'choice':
                # Step 1: Average output over the response window
                mean_outputs = np.mean(outputs[stim_on:, :, :], axis=0)  # shape: (n_trials, n_stims)
                # Step 2: Decode the choice (argmax across stimulus channels)
                choice_labels = np.argmax(mean_outputs, axis=1)  # shape: (n_trials,)
                tmp_over_acc[cv,:], tmp_stim_acc[cv,:,:] = decode_ls_svm(layer_data[l_num], s_label_int, n_afc, w_size, time_or_xgen, trn_prcnt)
        # average over cvs
        over_acc[m_num,l_num,:] = np.mean(tmp_over_acc,axis=0)
        stim_acc[m_num,l_num,:,:] = np.mean(tmp_stim_acc,axis=0)
        #print(f"done decoding {classes} for layer {l_num}")

#npz_name = f'plots/decoding/across_layers/{fn[15:-4]}0-{n_models -1}_ext_n{stim_noise}_int_n{noise_internal}.npz'
#np.savez( npz_name, over_acc = over_acc, stim_acc = stim_acc )
    
#--------------------------
# plot decoding stuff...figure out if time x time 
# or xgen, and if time x time if more than one model 
# in which case compute sem. 
#--------------------------
if time_or_xgen == 0: 
    
    
    
    # mean decoding and sem over models
    m_over_acc = np.nanmean(over_acc,axis=0)
    m_stim_acc = np.nanmean(stim_acc,axis=0)
    
    
    # Set grid size
    ncols = 3  # e.g., 3 columns per row
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))

    plt.rcParams.update({
        'font.size': 16,           # base font size
        'axes.titlesize': 18,      # subplot titles
        'axes.labelsize': 16,      # axis labels
        'xtick.labelsize': 16,     # x-axis tick labels
        'ytick.labelsize': 16,     # y-axis tick labels
        'legend.fontsize': 14,     # legend text
        'figure.titlesize': 18     # suptitle font
    })
        
    for layer in np.arange(n_layers):

        m_stim_acc = np.mean(stim_acc[:,layer, :,:], axis=0)
        sem_stim_acc = sem(stim_acc[:,layer, :,:], axis=0)
        t = np.arange(0, T, T / m_stim_acc.shape[1])
        
        # calculate AUC
        auc_expected = trapezoid(m_stim_acc[0,:], t)
        auc_unexpected = trapezoid(np.mean( m_stim_acc[1:,:],axis=0 ), t)

        ax = axes[layer]
        ax.errorbar( t, m_stim_acc[0,:],sem_stim_acc[0,:], fmt=hex_c[0], label = f'expected AUC: {auc_expected:.2f}' )
        ax.errorbar( t, np.mean( m_stim_acc[1:,:],axis=0 ), np.mean( sem_stim_acc[1:,:],axis=0 ), fmt=hex_c[1], label = f'unexpected AUC: {auc_unexpected:.2f}' )
        # for j in range(n_afc-1):
        #     ax.errorbar( t, m_stim_acc[j+1,:], sem_stim_acc[j+1,:], fmt=hex_c[1], label = f'unexpected {j+1}' )
        ax.set_title(f'layer {layer+1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Accuracy')
        # Per-plot legend with AUCs
        ax.legend(loc='lower right', frameon=False)
        ax.axvspan(stim_on, stim_on+stim_dur, alpha=0.2, color='gray')
        ax.axvspan(stim_on+stim_dur, T, alpha=0.2, color='beige')

        
    # Get handles and labels from one axis (e.g., the last one used)
    handles, labels = ax.get_legend_handles_labels()

    fig.suptitle(f'Decoding Accuracy {n_afc}-n_afc {int(stim_prob*100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)

    # Final layout tweaks
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for the legend

    # if task_type == 'rdk_reproduction':
    #     plt.savefig(f'plots/decoding/across_layers/auc_{file_path[35:-15]}.png')
    # elif task_type == 'rdk':
    #     plt.savefig(f'plots/decoding/across_layers/auc_{file_path[22:-15]}.png')
    # elif task_type == 'rdk_repro_cue':
        # plt.savefig(f'plots/across_layers/auc_{file_path[22:-15]}.png')

    plt.show()

        
   
# plot xgen
else:
    
    print('plotting cross generalization')

    # plot overall xgen
    # Set grid size
    ncols = 3  # e.g., 3 columns per row
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))

    plt.rcParams.update({
        'font.size': 16,           # base font size
        'axes.titlesize': 18,      # subplot titles
        'axes.labelsize': 16,      # axis labels
        'xtick.labelsize': 16,     # x-axis tick labels
        'ytick.labelsize': 16,     # y-axis tick labels
        'legend.fontsize': 14,     # legend text
        'figure.titlesize': 18     # suptitle font
    })
        
    for layer in np.arange(n_layers):

        m_over_acc = np.mean(over_acc[:,layer, :,:], axis=0)
        ax = axes[layer]
        ax.imshow(m_over_acc,origin='lower',aspect='equal')
        ax.set_title(f'layer {layer+1}')

    fig.suptitle(f'Decoding xgen {n_afc}-n_afc {int(stim_prob*100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)

    plt.show()
    
    # plot expected xgen
    # Set grid size
    ncols = 3  # e.g., 3 columns per row
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))

    plt.rcParams.update({
        'font.size': 16,           # base font size
        'axes.titlesize': 18,      # subplot titles
        'axes.labelsize': 16,      # axis labels
        'xtick.labelsize': 16,     # x-axis tick labels
        'ytick.labelsize': 16,     # y-axis tick labels
        'legend.fontsize': 14,     # legend text
        'figure.titlesize': 18     # suptitle font
    })
        
    for layer in np.arange(n_layers):

        m_stim_acc = np.mean(stim_acc[:,layer, 0,:], axis=0)
        ax = axes[layer]
        ax.imshow(m_stim_acc,origin='lower',aspect='equal')
        ax.set_title(f'layer {layer+1}')

    fig.suptitle(f'Decoding xgen expected {n_afc}-n_afc {int(stim_prob*100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)

    plt.show()
    
    # plot unexpected xgen
    # Set grid size
    ncols = 3  # e.g., 3 columns per row
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))

    plt.rcParams.update({
        'font.size': 16,           # base font size
        'axes.titlesize': 18,      # subplot titles
        'axes.labelsize': 16,      # axis labels
        'xtick.labelsize': 16,     # x-axis tick labels
        'ytick.labelsize': 16,     # y-axis tick labels
        'legend.fontsize': 14,     # legend text
        'figure.titlesize': 18     # suptitle font
    })
        
    for layer in np.arange(n_layers):

        m_stim_acc = np.mean(np.mean(stim_acc[:,layer, 1:,:], axis=0), axis =0)
        ax = axes[layer]
        ax.imshow(m_stim_acc,origin='lower',aspect='equal')
        ax.set_title(f'layer {layer+1}')

    fig.suptitle(f'Decoding xgen unexpected {n_afc}-n_afc {int(stim_prob*100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)

    plt.show()
    
    
# at the end remind me which one we were working on  
print(f'finished {settings}')
print('\007') # make a sound   