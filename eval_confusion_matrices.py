#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:47:27 2025

@author: hkular
"""

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
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
stim_prob_eval = 1/n_afc     
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
n_layers =3
batch_size = 2000
out_size = n_afc
time_or_xgen = 0
w_size = 5
classes = 'stim'
n_models = 20
fn_stem = f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer}/reprocue_'

#--------------------------
# Which conditions to compare
#--------------------------
cue_onsets = [0, 75]
cue_layer = 3
stim_probs = [1/n_afc, 0.7]
fb21_scalars = [1.0,0.7,0.3]
fb32_scalars = [1.0,0.7,0.3]
# valid_combos = [(1.0, 1.0)]  # always include both at 1.0
# # fb21 varies, fb32=1.0
# valid_combos += [(fb21, 1.0) for fb21 in fb21_scalars if fb21 != 1.0]
# # fb32 varies, fb21=1.0
# valid_combos += [(1.0, fb32) for fb32 in fb32_scalars if fb32 != 1.0]
valid_combos = list(product(fb21_scalars, fb32_scalars))

results = []

plots = False



# load model evals
for stim_prob in stim_probs:
    
    for cue_on in cue_onsets:
        
        
        for fb21_scalar, fb32_scalar in valid_combos:
    
             # load the correct model
             if stim_prob == 0.7:
                 fn = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob * 100)}_stim_prob_eval-{int(stim_prob_eval*100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
             else:
                 fn = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob * 100)}_stim_prob_eval-{int(stim_prob_eval*100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
             mod_data = np.load(fn, allow_pickle=True)
             print(f"Loaded {fn}, keys: {mod_data.files}")
             
             # loop over models
             for m_num in np.arange(n_models):
                
                 # load model
                 fn_mod = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob_train * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
                 
                 # load model sr_scram
                 with open(f'{fn_mod[:-3]}.json', "r") as infile:
                    _ , rnn_settings = json.load(infile)
                 sr_scram_list = rnn_settings['sr_scram']
                 sr_scram_list = [sr_scram_list[str(s)] for s in sorted(sr_scram_list.keys(), key=int)]
                 sr_scram = np.array(sr_scram_list)
                
                 # get labels
                 y_true = mod_data['stim_label'][m_num, stim_on,:].astype(int)
                 y_out = mod_data['outputs'][m_num,:]                 
                 y_pred = np.argmax(np.mean(y_out[stim_on:,:,:], axis = 0), axis = 1)
                 cues = np.argmax(mod_data['cues'][100, :,:], axis = 1)
                 # unscramble y_true
                 y_unscrambled = np.full((batch_size), np.nan)
                 for trial in np.arange(batch_size):
                     y_true_unscrambled[trial] = sr_scram[ cues[trial], y_true[trial] ].astype(int)
                
                 # Compute confusion matrix
                 cm = confusion_matrix(y_true, y_pred, labels=np.arange(6))
                
                 # Normalize (optional, per row)
                 cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            
                 results.append({
                    'stim_prob': int(100*stim_prob),
                    'cue_on': cue_on,
                    'cue_layer': cue_layer,
                    'model': m_num,
                    'fb21_scalar':fb21_scalar,
                    'fb32_scalar':fb32_scalar,
                    'cm_norm': cm_norm
                    })



fn_out = f"decode_data/plots/CM_{classes}_stimprob_x_cueon_cuelayer3_feedback.npz"

np.savez( fn_out,results = results)



if plots == True:
    # Plot heatmap
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Resp {i}" for i in range(6)],
                yticklabels=[f"Stim {i}" for i in range(6)])
    plt.xlabel("Model Response")
    plt.ylabel("True Stimulus")
    plt.show()