#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 12:21:22 2025

@author: hkular
"""

import numpy as np
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from scipy.stats import sem
from helper_funcs import *
from numpy import trapezoid

#--------------------------
# Basic model params
#--------------------------
task_type = 'rdk_repro_cue'                         # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                                           # number of stimulus alternatives
T = 210                                             # timesteps in each trial
cue_on = 75                                          # 0(start) or 75(stim offset)
cue_layer = 1                                      # which layer gets the cue
stim_prob_train = 1/n_afc
stim_prob_eval = stim_prob_train     
stim_amp_train = 1.0                                # can make this a list of amps and loop over... 
stim_amp_eval = 1.0
stim_noise_train = 0.1                              # magnitude of randn background noise in the stim channel
stim_noise_eval = 0.1
int_noise_train = 0.1                               # noise trained at 0.1
int_noise_eval = 0.1
weighted_loss = 0                                   # 0 = nw_mse l2 or 1 = weighted mse
num_cues = 2                                        # how many sr_scram
stim_on = 50                                        # timestep of stimulus onset
stim_dur = 25                                       # stim duration
cue_dur = T-cue_on                                  # on the rest of the trial
acc_amp_thresh = 0.8                                # to determine acc of model output: > acc_amp_thresh during target window is correct
h_size = 200                                        # how many units in a hidden layer
plots = False                                       # only plot if not run through terminal
n_layers =3
batch_size = 2000
out_size = n_afc

time_or_xgen = 1
w_size = 5
classes = 'stim'

settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_eval, 'stim_amp' : stim_amp_eval, 'stim_noise' : stim_noise_eval, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur}


# fn out for npz file to load decoding data
if time_or_xgen == 0:
    fn = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob_train * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse.npz'
else:
    fn = f'decode_data/{task_type}_xgen_{classes}_{n_afc}nafc_stim_prob{int(stim_prob_train * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse.npz'

# load the file
# load model and get relevant fields
mod_data = np.load( fn,allow_pickle=True )
# for varName in mod_data:
#         globals()[varName] = mod_data[varName]
over_acc = mod_data['over_acc']
stim_acc = mod_data['stim_acc']

# set up some nice colors (from Robert and Nuttida)
hex_c = ['#06D2AC', '#206975', '#6F3AA4', '#2B1644']

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
        ax.errorbar( t, m_stim_acc[0,:],sem_stim_acc[0,:], fmt=hex_c[0], label = f'cue_1 AUC: {auc_expected:.2f}' )
        ax.errorbar( t, np.mean( m_stim_acc[1:,:],axis=0 ), np.mean( sem_stim_acc[1:,:],axis=0 ), fmt=hex_c[1], label = f'cue_2 AUC: {auc_unexpected:.2f}' )
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
    
    fig.suptitle(f'Decoding Accuracy {n_afc}-n_afc {int(stim_prob_train*100)}-stim_prob cue_on{cue_on}', fontsize=16)
    
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
    
    fig.suptitle(f'Decoding xgen {n_afc}-n_afc {int(stim_prob_eval*100)}-stim_prob cue_on{cue_on}', fontsize=16)
    
    plt.show()
    
    # # plot expected xgen
    # # Set grid size
    # ncols = 3  # e.g., 3 columns per row
    # nrows = 1
    
    # fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    
    # plt.rcParams.update({
    #     'font.size': 16,           # base font size
    #     'axes.titlesize': 18,      # subplot titles
    #     'axes.labelsize': 16,      # axis labels
    #     'xtick.labelsize': 16,     # x-axis tick labels
    #     'ytick.labelsize': 16,     # y-axis tick labels
    #     'legend.fontsize': 14,     # legend text
    #     'figure.titlesize': 18     # suptitle font
    # })
        
    # for layer in np.arange(n_layers):
    
    #     m_stim_acc = np.mean(stim_acc[:,layer, 0,:], axis=0)
    #     ax = axes[layer]
    #     ax.imshow(m_stim_acc,origin='lower',aspect='equal')
    #     ax.set_title(f'layer {layer+1}')
    
    # fig.suptitle(f'Decoding xgen expected {n_afc}-n_afc {int(stim_prob_eval*100)}-stim_prob cue_on{cue_on}', fontsize=16)
    
    # plt.show()
    
    # # plot unexpected xgen
    # # Set grid size
    # ncols = 3  # e.g., 3 columns per row
    # nrows = 1
    
    # fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    
    # plt.rcParams.update({
    #     'font.size': 16,           # base font size
    #     'axes.titlesize': 18,      # subplot titles
    #     'axes.labelsize': 16,      # axis labels
    #     'xtick.labelsize': 16,     # x-axis tick labels
    #     'ytick.labelsize': 16,     # y-axis tick labels
    #     'legend.fontsize': 14,     # legend text
    #     'figure.titlesize': 18     # suptitle font
    # })
        
    # for layer in np.arange(n_layers):
    
    #     m_stim_acc = np.mean(np.mean(stim_acc[:,layer, 1:,:], axis=0), axis =0)
    #     ax = axes[layer]
    #     ax.imshow(m_stim_acc,origin='lower',aspect='equal')
    #     ax.set_title(f'layer {layer+1}')
    
    # fig.suptitle(f'Decoding xgen unexpected {n_afc}-n_afc {int(stim_prob_eval*100)}-stim_prob cue_on{cue_on}', fontsize=16)
    
    # plt.show()


