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
import pandas as pd
import seaborn as sns
from rdk_task import RDKtask
import torch
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
batch_size = 2400
out_size = n_afc
time_or_xgen = 0
w_size = 5
classes = 'stim'
n_models = 20
rand_seed_bool = True
seed_num=42
equal_balance = True
fn_stem = f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer}/reprocue_'
settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_eval, 'stim_amp' : stim_amp_eval, 'stim_noise' : stim_noise_eval, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur, 'rand_seed_bool':rand_seed_bool, 'seed_num':seed_num}

# create the task object
task = RDKtask( settings )
#--------------------------
# Which conditions to compare
#--------------------------
cue_onsets = [0, 75]
cue_layer = 3
stim_probs = [ 0.7]
fb21_scalars = [1.0,0.7]
fb32_scalars = [1.0,0.7]
valid_combos = list(product(fb21_scalars, fb32_scalars))

results = []

plots = False
err_types = {"no_response": 0, "sub_thresh_response":0, "multiple_response": 0, "wrong_response": 0}

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
                 
                 if task_type == 'rdk_repro_cue':
                     # load model sr_scram
                     with open(f'{fn_mod[:-3]}.json', "r") as infile:
                        _ , rnn_settings = json.load(infile)
                     sr_scram_list = rnn_settings['sr_scram']
                     sr_scram_list = [sr_scram_list[str(s)] for s in sorted(sr_scram_list.keys(), key=int)]
                     sr_scram = np.array(sr_scram_list)
                     
                     cues = mod_data['cues'][m_num,:]
                     c_label = np.array(cues, dtype=int)
                     
                 else:
                     sr_scram = []
                    
                 # get labels
                 y_true = mod_data['stim_label'][m_num, stim_on,:].astype(int)
                 y_out = mod_data['outputs'][m_num,:]         
                 outputs = torch.from_numpy(y_out)
                 s_label = np.zeros((batch_size, n_afc), dtype=int)
                 s_label[np.arange(batch_size), y_true] = 1
                 m_acc = mod_data['m_acc'][m_num]
                 tbt_acc = mod_data['tbt_acc'][m_num,:]
                 
       
                 y_pred = np.full((batch_size),np.nan)
                 y_guess = np.full((batch_size),np.nan)
                 for nt in range(batch_size):
                                      
                     if tbt_acc[nt] == 1:
                         y_pred[nt] = y_true[nt]
                         y_guess[nt] = -6                          
                     else:
                         non_targ = np.setdiff1d(np.arange(6), y_true[nt]) 
                         # classify errors
                         above = np.where(np.mean(y_out[stim_on:,nt,:], axis = 0) >= acc_amp_thresh)[0]
                         below = np.where(np.mean(y_out[stim_on:,nt,:], axis = 0) >= 0.65)[0]
                         

                         if len(above) == 0:
                             y_pred[nt] = 6# no response
                             if len(below)==0:
                                 y_guess[nt] = -1
                                 err_types["no_response"] += 1
                             else:
                                 y_guess[nt] = np.argmax(np.mean(y_out[stim_on:,nt,non_targ], axis = 0), axis = 0)
                                 err_types["sub_thresh_response"]+=1
                         elif len(above) > 2:
                             y_guess[nt] = np.argmax(np.mean(y_out[stim_on:,nt,non_targ], axis = 0), axis = 0)
                             y_pred[nt] = 7  # multiple response                             
                             err_types["multiple_response"] += 1
                         else:
                             y_guess[nt] = np.argmax(np.mean(y_out[stim_on:,nt,non_targ], axis = 0), axis = 0)
                             y_pred[nt] = 8  # wrong single response
                             err_types["wrong_response"] += 1



                 # # y_pred = np.argmax(np.mean(y_out[stim_on:,:,:], axis = 0), axis = 1)
                 
                 # # Step 2. Unscramble y_pred by inverting the mapping
                 # # For each cue, invert the mapping once
                 # unscramble = np.zeros_like(sr_scram)
                 # for c in range(sr_scram.shape[0]):
                 #     for stim in range(sr_scram.shape[1]):
                 #         scrambled_label = sr_scram[c, stim]
                 #         unscramble[c, scrambled_label] = stim
                
                 # # Step 3. Apply unscramble to model predictions
                 # y_pred_unscrambled = np.array([unscramble[c, p] for c, p in zip(c_label, y_pred.astype(int))])
 
                 # cm_acc = (y_true == y_pred).sum()/batch_size
                 # print(f'eval acc: {mod_data['m_acc'][m_num]}, cm acc: {cm_acc}')
                
                 # # Compute confusion matrix
                 # cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
                
                 # # Normalize (optional, per row)
                 # row_sums = cm.sum(axis=1)[:, np.newaxis]
                 # cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums!=0)
                                  
                 vals, counts = np.unique(y_guess, return_counts=True)

                 # Filter only values >= -1
                 mask = vals >= -1
                 vals = vals[mask].astype(int)      # ensure integers for dict keys
                 counts = counts[mask]
                
                 # Turn into dictionary {value: count}
                 guess_types = dict(zip(vals, counts))
            
                 results.append({
                    'stim_prob': int(100*stim_prob),
                    'cue_on': cue_on,
                    'cue_layer': cue_layer,
                    'model': m_num,
                    'fb21_scalar':fb21_scalar,
                    'fb32_scalar':fb32_scalar,
                    'eval_acc': mod_data['m_acc'][m_num],
                    'guess_types': guess_types, 
                    'err_types': err_types
                    })



fn_out = f"decode_data/plots/CM_{classes}_{task}_feedback.npz"

np.savez( fn_out,results = results, allow_pickle = True)



if plots == True:
    
    
    data = np.load(f"decode_data/plots/CM_{classes}_{task}_feedback.npz", allow_pickle = True)
    results = data['results']
    results_list = [item for item in results]  # Convert back to list
    df = pd.DataFrame(results_list)
    #df = pd.DataFrame(data['results'])
    df['cue_layer'] = df['cue_layer'].astype(str)
    df['stim_prob'] = df['stim_prob'].replace({16: 'Unbiased', 70: 'Biased'})
    cueon_map = {0: 'Start', 75: 'Stim Offset'}
    df['cue_on'] = df['cue_on'].map(cueon_map)
    df['cue_on'] = pd.Categorical(
        df['cue_on'],
        categories=['Start', 'Stim Offset'],
        ordered=True
    )
    
    
    # subset data
    df_ex = df[(df['stim_prob']=='Unbiased') &
               (df['fb21_scalar']==1.0) &
               (df['fb32_scalar']==1.0) &
               (df['cue_on']=='Start') &
               (df['model']==19) &
               (df['cue_layer']=='3')]
    
    # Plot heatmap
    sns.heatmap(df_ex['cm_norm'].iloc[0], annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Resp {i}" for i in range(6)],
                yticklabels=[f"Stim {i}" for i in range(6)])
    plt.xlabel("Model Response")
    plt.ylabel("True Stimulus")
    plt.show()