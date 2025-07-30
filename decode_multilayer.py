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

# set up some nice colors (from Robert and Nuttida)
hex_c = ['#06D2AC', '#206975', '#6F3AA4', '#2B1644']

#--------------------------
# Basic model params
#--------------------------
device = 'cpu'                    # device to use for loading/eval model
task_type = 'rdk_repro_cue'                 # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                       # number of stimulus alternatives
T = 200                           # timesteps in each trial
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_prob = 1/n_afc     # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
stim_prob_to_eval = stim_prob           # eval the model at this prob level (stim_prob is used to determine which trained model to use)
stim_amps_train = 0.6                # can make this a list of amps and loop over... 
stim_amps = stim_amps_train
stim_noise_train = 0.1
stim_noise = stim_noise_train                 # magnitude of randn background noise in the stim channel
batch_size = 3000                 # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct
weighted_loss = 0                       #  0 = nw_mse l2 or 1 = weighted mse
noise_internal = 0.1            # noise trained at 0.1
num_cues = 2
cue_on = 100
cue_dur = 150

if task_type == 'rdk':
    fn_stem = 'trained_models_rdk/gonogo_'
    out_size = 1
elif task_type == 'rdk_reproduction':
    fn_stem = 'trained_models_rdk_reproduction/repro_'
    out_size = n_afc  
elif task_type == 'rdk_repro_cue':
    fn_stem = 'trained_models_rdk_repro_cue/reprocue_'
    out_size = n_afc  



#--------------------------
# decoding params
#--------------------------
trn_prcnt = 0.8    # percent of data to use for training
n_trn_tst_cvs = 5  # how many train/test cv folds
time_or_xgen = 0   # decode timepnt x timepnt or do full xgen matrix 
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
n_models = 5
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
    outputs,s_label,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3,m_acc,tbt_acc = eval_model( net, task, sr_scram )
    
    # s_label is a diff shape for cue version, deal with if statement later
    s_label = np.argmax(s_label, axis=1)
    
    # quick plots - first expected, correct and incorrect trials
    plt.plot(np.squeeze( outputs[ :,(s_label==0) & (tbt_acc==1),0 ] ), c=hex_c[0], alpha=0.25 )
    plt.plot(np.squeeze( outputs[ :,(s_label==0) & (tbt_acc==0),0 ] ), c=hex_c[1], alpha=0.25 )
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    plt.title(f'expected {n_afc}-afc {stim_amps}-amp {weighted_loss}-weighted_loss mod-{m_num}')
    plt.show()
    
    # quick plots - the unexpected, correct and incorrect trials
    blues = plt.cm.Blues(np.linspace(0.4, 0.9, out_size))  # Skip very light colors
    reds = plt.cm.Reds(np.linspace(0.4, 0.9, out_size))
    # Convert RGBA to hex
    hex_c_correct = [matplotlib.colors.rgb2hex(c[:3]) for c in blues]
    hex_c_incorrect = [matplotlib.colors.rgb2hex(c[:3]) for c in reds]
    for i in np.arange(out_size):
        plt.plot(np.squeeze( outputs[ :,(s_label!=0) & (tbt_acc==1),i ] ), c=hex_c_correct[i], alpha=0.25 )
        plt.plot(np.squeeze( outputs[ :,(s_label!=0) & (tbt_acc==0),i ] ), c=hex_c_incorrect[i], alpha=0.25 )
        plt.xlabel('Time Step')
        plt.ylabel('Output')
    #plt.ylim((-0.2,1.1))
    plt.title(f'unexpected {n_afc}-afc {stim_amps}-amp {weighted_loss}-weighted_loss mod-{m_num}')
    plt.show()
    

    #  decoding
    for l_num in np.arange(n_layers):
        over_acc[m_num,l_num, :], stim_acc[m_num, l_num,:,:] = decode_svc(stim_prob, h1, s_label, trn_prcnt, n_trn_tst_cvs, time_or_xgen, w_size, num_cs, n_cvs_for_grid, max_iter)
        

#npz_name = f'plots/decoding/across_layers/{fn[15:-4]}0-{n_models -1}_ext_n{stim_noise}_int_n{noise_internal}.npz'
#np.savez( npz_name, over_acc = over_acc, stim_acc = stim_acc )
    
#--------------------------
# plot decoding stuff...figure out if time x time 
# or xgen, and if time x time if more than one model 
# in which case compute sem. 
#--------------------------
if time_or_xgen == 0: 
    
    
    
    # mean decoding and sem over models
    m_over_acc = np.mean(over_acc,axis=0)
    m_stim_acc = np.mean(stim_acc,axis=0)
    
    
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
        ax.set_title(f'layer {i+1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Accuracy')
        # Per-plot legend with AUCs
        ax.legend(loc='lower right', frameon=False)
        ax.axvspan(stim_on, stim_on+stim_dur, alpha=0.2, color='gray')

        
    # Get handles and labels from one axis (e.g., the last one used)
    handles, labels = ax.get_legend_handles_labels()

    #fig.suptitle(f'Decoding Accuracy {n_afc}-n_afc {int(stim_prob*100)}-stim_prob {stim_amps}-stim_amp {stim_noise}-stim_noise', fontsize=16)

    # Final layout tweaks
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for the legend

    # if task_type == 'rdk_reproduction':
    #     plt.savefig(f'plots/decoding/across_layers/auc_{file_path[35:-15]}.png')
    # elif task_type == 'rdk':
    #     plt.savefig(f'plots/decoding/across_layers/auc_{file_path[22:-15]}.png')
        

    plt.show()

        
   
# plot xgen
else:
    
    print('plotting cross generalization')
    
    # plot mean decoding over models
    m_over_acc = np.mean(over_acc,axis=0)
    m_stim_acc = np.mean(stim_acc,axis=0)
    
    # overall acc
    plt.imshow(m_over_acc,origin='lower',aspect='equal')
    plt.colorbar()
    plt.title(f'{n_afc}-afc {stim_amps}-amp {weighted_loss}-weighted_loss overall')
    plt.show()
    
    # for each stim - first expected
    plt.imshow(m_stim_acc[0,:,:],origin='lower',aspect='equal')
    plt.colorbar()
    plt.title(f'{n_afc}-afc {stim_amps}s-amp {weighted_loss}-weighted_loss expected')
    plt.show()    
    
    # for each stim - then avg of unexpected
    plt.imshow(np.mean(m_stim_acc[1:,:,:],axis=0),origin='lower',aspect='equal')
    plt.colorbar()
    plt.title(f'{n_afc}-afc {stim_amps}-amp {weighted_loss}-weighted_loss unexpected')
    plt.show()        
    
    
# at the end remind me which one we were working on  
print(f'finished {settings}')
print('\007') # make a sound   