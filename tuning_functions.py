#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:22:25 2025

@author: hkular
"""

# tuning analysis

# calculate delta between pre-sitm period and the stim-period for each unit
# sum abs deltas for each stim
# look at sum(deltas[deltas>0])
# delta for each unit and then sort units based on delta, plot as line, tuning function for each stim


import numpy as np
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from helper_funcs import *
import seaborn as sns
import pandas as pd
from itertools import product
from rdk_task import RDKtask
import torch
import json

#--------------------------
# Basic model params
#--------------------------
task_type = 'rdk_repro_cue'                         # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                                           # number of stimulus alternatives
T = 210                                             # timesteps in each trial
cue_on = 75                                          # 0(start) or 75(stim offset)
cue_layer = 3                                      # which layer gets the cue
stim_prob_train = 0.7
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
batch_size = 1000
out_size = n_afc
device = 'cpu'
time_or_xgen = 0
w_size = 5
classes = 'stim'
fn_stem = f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer3/reprocue_'
settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_eval, 'stim_amp' : stim_amp_eval, 'stim_noise' : stim_noise_eval, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur}
# create the task object
task = RDKtask( settings )



# metrics
n_models = 20
n_stim_types = n_afc
decay_window = 50
sustain_window = 50
stim_offset = stim_on+stim_dur
decay_rates = np.zeros((n_models, n_layers, n_stim_types))
sustained_acc = np.zeros((n_models, n_layers, n_stim_types))

#--------------------------
# Which conditions to compare
#--------------------------
cue_onsets = [0, 75]
cue_layer = 3
stim_probs = [1/n_afc, 0.7]
valid_combos = []

results = []

plots = False

n_models = 1#count_models(n_afc, stim_prob, stim_amps_train, stim_noise_train, weighted_loss, task_type, fn_stem, directory = os.getcwd())

         
# timing
pre_stim_period = np.arange(0,stim_on)
stim_period = np.arange(stim_on,stim_offset)
             
for stim_prob in stim_probs:
    for cue_on in cue_onsets:
        for m_num in range(n_models):
            
            # build a file name...
            if weighted_loss == 0:
                fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
            else:
                # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
                if stim_prob == 1 / n_afc:
                    fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'
                else:
                    fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'  

                
            # load the trained model, set to eval, requires_grad == False
            if device == 'cuda':
                net = load_model( fn, cuda_num )
            elif device == 'cpu':   
                net = load_model( fn,device )
            print(f'loaded model {m_num}')
            
            # load cue scramble matrix
            with open(f'{fn[:-3]}.json', "r") as infile:
               _ , rnn_settings = json.load(infile)
            sr_scram_list = rnn_settings['sr_scram']
            sr_scram_list = [sr_scram_list[str(s)] for s in sorted(sr_scram_list.keys(), key=int)]
            sr_scram = np.array(sr_scram_list)
          
            # load the correct model        
            # eval a batch of trials using the trained model
            _,s_label,_,_,_,_,_,_,_,_,_,h1,h2,h3,_,_,_,_,_,_,_,_,_,_ = eval_model( net, task, sr_scram )
            s_label_int = np.argmax(s_label, axis=1)
             
            # layers
            layer_data = [h1, h2, h3] # time, trials, units
           
            
            for l in range(n_layers):                
                for stim in range(n_afc):
                    
                    # average over trials for that stim
                    stim_data = np.mean(layer_data[l][:,s_label_int==stim,:], axis=1) 
                    
                    # subtract average over time for that stim during stim_on and pre-stim
                    delta = np.mean(stim_data[stim_period,:],axis=0) - np.mean(stim_data[pre_stim_period,:],axis=0)
                    
                    # sort based on delta
                    sorted_idx = np.argsort(delta, axis=0)
                    delta_sorted = delta[sorted_idx]
                   
                    sum_deltas = sum(abs(delta))
                    
                    results.append({
                       'stim_prob': int(100*stim_prob),
                       'cue_on': cue_on,
                       'cue_layer': cue_layer,
                       'model': m_num,
                       'layer': l+1,
                       'stim': stim,
                       'sum_deltas': sum_deltas,
                       'sorted_deltas': delta_sorted,
                       'stim_data': stim_data,
                       'delta_per_unit': np.atleast_1d(delta)
                   
                       })


# plot tuning curves for each layer

# Convert results list to DataFrame
df = pd.DataFrame(results)

layers = sorted(df['layer'].unique())
stims = sorted(df['stim'].unique())

for stim_prob in stim_probs:
    
    df_ex = df[df['stim_prob']==int(100*stim_prob)]
    
    # Create a figure with 3 panels, one per layer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for i, l in enumerate(layers):
        ax = axes[i]
        df_layer = df_ex[df_ex['layer'] == l]
        
        for stim in stims:
            delta_sorted = df_layer[df_layer['stim'] == stim]['sorted_deltas'].values[0]
            ax.plot(delta_sorted, label=f'Stim {stim}')
        
        ax.set_title(f'Layer {l}')
        ax.set_xlabel('Unit rank')
        if i == 0:
            ax.set_ylabel('Delta firing rate')
        ax.legend()
    
    plt.suptitle(f'Tuning function by layer stim prob {int(100*stim_prob)}')
    plt.tight_layout()
    plt.show()




# plot delta FR over stim
# Build per-layer, per-unit tuning curves
# for l in range(n_layers):
#     layer_results = [r for r in results if r['layer'] == l+1]
#     n_units = len(layer_results[0]['delta_per_unit'])
#     n_stim = n_afc
    
#     unit_deltas = np.zeros((n_units, n_stim))
#     for r in layer_results:
#         unit_deltas[:, r['stim']] = r['delta_per_unit']
    
#     # plot per-unit tuning curves
#     for u in range(n_units):
#         plt.plot(range(n_stim), unit_deltas[u, :], marker='o', alpha=0.5)
#     plt.title(f'Layer {l+1} - Per-unit tuning curves')
#     plt.xlabel('Stimulus')
#     plt.ylabel('Delta firing rate')
#     plt.show()


# # plot for each stim

# for s in range(n_afc):
#     stim_layer_results = [r for r in results if (r['layer'] == 1) & (r['stim']==s)]
#     n_units = len(stim_layer_results[0]['delta_per_unit'])
#     # plot per-unit tuning curves
#     plt.plot( r['delta_per_unit'], marker='o', alpha=0.5)
#     plt.title(f'Layer 1 - stim {s} tuning function')
#     #plt.xlabel('Stimulus')
#     plt.ylabel('Delta firing rate')
#     plt.show()






# # plot firing rate as a function of stim -- work on this
# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# for i, l in enumerate(layers):
#     ax = axes[i]
#     df_layer = df[df['layer'] == l]
    
#     sns.heatmap(tuning_curves, cmap='viridis', cbar=True, xticklabels=range(afc), ax=axs[0])
#     #for stim in stims:
#         #delta_sorted = df_layer[df_layer['stim'] == stim]['sorted_deltas'].values[0]
#         #ax.plot(delta_sorted, label=f'Stim {stim}')
    
#     ax.set_title(f'Layer {l}')
#     ax.set_xlabel('Stimulus')
#     if i == 0:
#         ax.set_ylabel('Delta firing rate')
#     ax.legend()

# plt.suptitle('Tuning curves by layer')
# plt.tight_layout()
# plt.show()