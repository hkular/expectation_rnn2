#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:12:06 2025

@author: hkular

comparing confusion matrices across model conditions
"""

#--------------------------
# imports
#--------------------------
import numpy as np
import torch
import json
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from scipy.stats import sem
from helper_funcs import *
from rdk_task import RDKtask
from model_count import count_models
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#--------------------------
# Basic model params
#--------------------------

device = 'cpu'                    # device to use for loading/eval model
task_type = 'rdk_repro_cue'                 # task type (conceptually think of as a motion discrimination task...)         
T = 225                          # timesteps in each trial
n_afc = 6
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_prob = 0.8           # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
stim_prob_to_eval = 1/n_afc          # eval the model at this prob level (stim_prob is used to determine which trained model to use)
stim_amps_train = 1.0                # can make this a list of amps and loop over... 
stim_amps = 1.0
stim_noise_train = 0.1
stim_noise = stim_noise_train                 # magnitude of randn background noise in the stim channel
batch_size = 1000                 # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct
weighted_loss = 0                       #  0 = nw_mse l2 or 1 = weighted mse
num_cues = 2
cue_on = 75
cue_dur = T-cue_on
cue_layer = 3
# %%



if task_type == 'rdk':
    fn_stem = 'trained_models_rdk/gonogo_'
elif task_type == 'rdk_reproduction':
    fn_stem = 'trained_models_rdk_reproduction/repro_'
elif task_type == 'rdk_repro_cue':
    fn_stem =  f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer}/reprocue_'

# Preset some conditions
afcs = [6]
amps = [1.0]
results = []

#--------------------------
# loop over model types
#--------------------------
for n_afc in afcs:
    
    # set ouput size based on this n_afc
    if task_type == 'rdk':
        out_size = 1
    else:
        out_size = n_afc
    # set probs based on this n_afc
    stim_probs = [0.8]
    
    for stim_prob in stim_probs:
        
        # set stim_prob at eval based on this stim_prob
        stim_prob_to_eval = 1/n_afc
    
    
        for stim_amps in amps:
            
            
            #--------------------------
            # init dict of task related params
            # note that stim_prob_to_eval is passed in here
            # and that the model name (fn) will be generated based 
            # on stim_prob... 
            #--------------------------
            settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
                        'stim_prob' : stim_prob_to_eval, 'stim_amp' : stim_amps, 'stim_noise' : stim_noise, 'batch_size' : batch_size, 
                        'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':150}
        
            # create the task object based on unique settings
            task = RDKtask( settings )
                      
            
            # Look up how many models to run for this condition
            n_models = count_models(n_afc, stim_prob, stim_amps_train, stim_noise_train, weighted_loss, task_type, fn_stem, directory = os.getcwd())
        
            #--------------------------
            # loop over trained models 
            #--------------------------
            conf_matrices = []
            conf_sems = []
            model_accuracies = []
            model_sems = []
            
            for m_num in np.arange( n_models ):
            
                # build a file name...
                if weighted_loss == 0:
                    fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
                else:
                    # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
                    if stim_prob == 1 / n_afc:
                        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'
                    else:
                        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'  
                
        
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
             
                # eval a batch of trials using the trained model
                outputs,s_label,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau1,tau3,m_acc,tbt_acc,cues = eval_model( net, task, sr_scram )
                if task_type == 'rdk_reproduction':
                    cm, _ = confusion_matrix_reproduction( outputs, s_label, settings )
                elif task_type == 'rdk_repro_cue':
                    s_label = np.argmax(s_label, axis=1)
                    cm, _ = confusion_matrix_reproduction( outputs, s_label, settings )
                elif task_type == 'rdk':
                    cm, _ = confusion_matrix_rdk( outputs, s_label, settings )
                    
                
                conf_matrices.append(cm)
                model_accuracies.append(m_acc)
            
            # compute average across models
            mean_cm = np.mean(conf_matrices,axis=0)
            sem_cm = sem(conf_matrices, axis = 0)
            mean_m_acc = np.mean(model_accuracies, axis = 0)
            sem_m_acc = sem(model_accuracies, axis = 0)
    
            # Save average across models of this type
            results.append({
                'afc': n_afc,
                'stim_prob': stim_prob,
                'stim_amp': stim_amps,
                'model_idx': m_num,
                'm_acc': mean_m_acc,
                'sem_acc': sem_m_acc,
                'confusion_matrix': mean_cm,
                'sem_cm': sem_cm
            })
        
        

print('\007') # make a sound 

  
# In[]
import pandas as pd
import seaborn as sns

# convert list to dataframe
df = pd.DataFrame(results)
# Define balanced/unbalanced
df['stim_prob'] = df['stim_prob'].apply(lambda x: 'unbiased' if np.isclose(x, 1/6) or np.isclose(x, 1/3) else 'biased')
# Set afc as categorical (ensures correct spacing)
df['afc'] = pd.Categorical(df['afc'], categories=sorted(df['afc'].unique()))

# save df
# df.to_csv(f'plots/compare_evals/{task_type}/stim{int( stim_noise * 100 )}_nw_mse_CM.csv', index=False)


##### plot confusion matrices

# Get unique combinations
unique_conditions = df[['afc', 'stim_prob', 'stim_amp']].drop_duplicates()
n_conditions = len(unique_conditions)

# Set grid size
ncols = 4  # e.g., 3 columns per row
nrows = int(np.ceil(n_conditions / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

# Flatten axes array for easy indexing
axes = axes.flatten()


for i, (_, row) in enumerate(unique_conditions.iterrows()):
    # Filter for this condition
    subset = df[
        (df['afc'] == row['afc']) & 
        (df['stim_prob'] == row['stim_prob']) & 
        (df['stim_amp'] == row['stim_amp'])
    ]
    
    if subset.empty:
        continue
    
    # Extract confusion matrix and plot it
    cm = subset.iloc[0]['confusion_matrix']
    cm_mod = np.delete(cm, -1, axis=0)
    ax = axes[i]
    im = ax.imshow(cm_mod, cmap='Blues')
    
    n_cols = cm_mod.shape[1]
    ax.axvline(x=n_cols - 1.5, color='r', linewidth=2, linestyle = '--')


    ax.set_title(f'P={row["stim_prob"]}, Cue_on{cue_on}, Cue_L{cue_layer}, n_mods{m_num+1}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Optionally annotate cells
    for (j, k), val in np.ndenumerate(cm_mod):
        ax.text(k, j, f'{val:.2f}', ha='center', va='center', fontsize=8)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#### plot mean accuracies

# # take a look to see if observations for each model is represented should be 2^3 = 8 rows
# df.groupby(['afc', 'stim_prob', 'stim_amp']).size().reset_index().rename(columns={0:'count'})



# # Define balanced/unbalanced
# df['train_type'] = df['stim_prob'].apply(lambda x: 'balanced' if np.isclose(x, 1/6) or np.isclose(x, 1/3) else 'unbalanced')
# # Set afc as categorical (ensures correct spacing)
# df['afc'] = pd.Categorical(df['afc'], categories=sorted(df['afc'].unique()))


# #  plot balanced training
# # Step 1: Compute model-level averages
# model_avg = df.groupby(['afc', 'train_type', 'stim_amp', 'model_idx'])['m_acc'].mean().reset_index()

# # Step 2: Compute mean and SEM per condition
# summary = model_avg.groupby(['afc', 'train_type', 'stim_amp']).agg(
#     mean_acc=('m_acc', 'mean'),
#     sem_acc=('m_acc', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
# ).reset_index()

# # Step 3: Plot
# g = sns.FacetGrid(model_avg, col='stim_amp', sharey=True, height=4, aspect=1.2)

# # Plot the individual model scatter points
# g.map_dataframe(
#     sns.stripplot,
#     x='afc',
#     y='m_acc',
#     hue='train_type',
#     dodge=True,
#     jitter=True,
#     alpha=0.5
# )

# # Overlay mean ± SEM
# for ax, stim_amp_val in zip(g.axes.flat, g.col_names):
#     for train_type in summary['train_type'].unique():
#         for afc_val in df['afc'].cat.categories:
#             subset = summary[
#                 (summary['stim_amp'] == stim_amp_val) &
#                 (summary['train_type'] == train_type) &
#                 (summary['afc'] == afc_val)
#             ]
#             if not subset.empty:
#                 x_pos = df['afc'].cat.categories.get_loc(afc_val)
#                 offset = -0.2 if train_type == 'balanced' else 0.2
#                 ax.errorbar(
#                     x=x_pos + offset,
#                     y=subset['mean_acc'].values[0],
#                     yerr=subset['sem_acc'].values[0],
#                     fmt='o',
#                     capsize=4,
#                     markersize=6,
#                 )

# # Adjust legends to remove duplicates
# handles, labels = g.axes[0][0].get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# g.axes[0][0].legend(by_label.values(), by_label.keys(), title='Training')

# # Final touches
# g.set_axis_labels("AFC", "Performance (%)")
# g.set_titles("Stim Amp: {col_name}")
# #g.add_legend(title='Training')
# plt.tight_layout()
# plt.show()


