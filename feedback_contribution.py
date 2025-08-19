#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:38:10 2025

@author: hollykular
"""

#--------------------------
# imports
#--------------------------
import numpy as np
import json
import os
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from helper_funcs import *
from rdk_task import RDKtask
from model_count import count_models
import argparse
import pandas as pd


#--------------------------
# setup argparser
# #--------------------------
parser = argparse.ArgumentParser(description='Analyze RNNs')
# decoding params
# parser.add_argument('--gpu', required=False,
#         default='0', help="Which gpu?")
# parser.add_argument('--device', required=False,
#         default='cpu', help="gpu or cpu?")
# parser.add_argument('--classes', required=False,
#         default='stim', help="stim or choice or cue?")
# parser.add_argument('--time_or_xgen', required=True,
#                     type=int, help="time=0, xgen=1")
# # model params
# parser.add_argument('--task_type', required=False,type=str,
#         default='rdk_repro_cue', help="Which task for train and eval?")
# parser.add_argument('--T', required=False, type=int,
#         default='210', help="How long is the trial?")
# parser.add_argument('--cue_on', required=True, type=int,
#         help="When does cue come on?")
# parser.add_argument('--cue_layer', required=True, type=int,
#         help="Which layer receives the cue?")
# parser.add_argument('--n_afc', required=False,type=int,
#         default=6,help="How many stimulus alternatives?")
# parser.add_argument('--stim_prob_train', required=True,type=int,
#         help="What stim prob during training?")
# parser.add_argument('--stim_prob_eval', required=True,type=int,
#         help="What stim prob during eval?")
# parser.add_argument('--stim_noise_train', required=True,type=float,
#         help="What stim noise during training?")
# parser.add_argument('--stim_noise_eval', required=True,type=float,
#         help="What stim noise during eval?")
# parser.add_argument('--stim_amp_train', required=True,type=float,
#         help="What stim amp during training?")
# parser.add_argument('--stim_amp_eval', required=True,type=float,
#         help="What stim amp during eval?")
# parser.add_argument('--int_noise_train', required=False,type=float,
#         default=0.1,help="What internal noise during training?")
# parser.add_argument('--int_noise_eval', required=False,type=float,
#         default=0.1, help="What internal noise during eval?")
# parser.add_argument('--batch_size', required=False,type=int, default = 2000,
#         help="How many trials to eval?")
# parser.add_argument('--weighted_loss', required=False,type=int, default = 0,
#         help="Weighted loss (1) or unweighted loss(0)?")



args = parser.parse_args()

# for debugging
args.task_type = 'rdk_repro_cue'
args.gpu = 1
args.device = 'cpu'
args.classes = 'cue'
args.time_or_xgen = 0
args.n_afc = 6
args.T = 210
args.cue_on = 75
args.cue_layer = 3
args.stim_prob_train = 1 / args.n_afc
args.stim_prob_eval = 1 / args.n_afc
args.stim_noise_train = 0.1
args.stim_noise_eval = 0.1
args.stim_amp_train = 1.0
args.stim_amp_eval = 1.0
args.int_noise_train = 0.1
args.int_noise_eval = 0.1
args.batch_size = 1000
args.weighted_loss = 0


#--------------------------
# Basic model params
#--------------------------
device = args.device                                # device to use for loading/eval model
task_type = args.task_type                          # task type (conceptually think of as a motion discrimination task...)         
n_afc = args.n_afc                                  # number of stimulus alternatives
T = args.T                                          # timesteps in each trial
cue_on = args.cue_on                                # 0(start) or 75(stim offset)
cue_layer = args.cue_layer                          # which layer gets the cue
if args.stim_prob_train > 1/n_afc:
    stim_prob_train = args.stim_prob_train /100     # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
else:
    stim_prob_train = 1/n_afc
if args.stim_prob_eval > 1/n_afc:
    stim_prob_eval = args.stim_prob_train /100     # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
else:
    stim_prob_eval = 1/n_afc        
stim_amp_train = args.stim_amp_train                # can make this a list of amps and loop over... 
stim_amp_eval = args.stim_amp_eval
stim_noise_train = args.stim_noise_train            # magnitude of randn background noise in the stim channel
stim_noise_eval = args.stim_noise_eval
int_noise_train = args.int_noise_train              # noise trained at 0.1
int_noise_eval = args.int_noise_eval
batch_size = args.batch_size                        # number of trials in each eval batch
weighted_loss = args.weighted_loss                  # 0 = nw_mse l2 or 1 = weighted mse
num_cues = 2                                        # how many sr_scram
stim_on = 50                                        # timestep of stimulus onset
stim_dur = 25                                       # stim duration
cue_dur = T-cue_on                                  # on the rest of the trial
acc_amp_thresh = 0.8                                # to determine acc of model output: > acc_amp_thresh during target window is correct
h_size = 200                                        # how many units in a hidden layer
plots = False                                      # only plot if not run through terminal
iters = 200000                                      # cut off during training

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
trn_prcnt = 0.8                     # percent of data to use for training
n_cvs = 5                           # how many train/test cv folds
classes = args.classes              # which are we decoding stim or choice
time_or_xgen = args.time_or_xgen    # decode timepnt x timepnt or do full xgen matrix 
w_size = 5                          # mv_avg window size
num_cs = 1                          # number of C's to grid search, if 1 then C=1
n_cvs_for_grid = 5                  # num cv folds of training data to find best C
max_iter = 5000                     # max iterations

#--------------------------
# init dict of task related params
# note that stim_prob_eval is passed in here
# and that the model name (fn) will be generated based 
# on stim_prob... 
#--------------------------
settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_eval, 'stim_amp' : stim_amp_eval, 'stim_noise' : stim_noise_eval, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur}

# create the task object
task = RDKtask( settings )


#--------------------------
# How many trained models in this cond
#--------------------------
n_models = count_models(n_afc, stim_prob_train, stim_amp_train, stim_noise_train, weighted_loss, task_type, fn_stem, directory = os.getcwd())
n_layers = 3

#--------------------------
# matrices to store model fb contributions
#--------------------------

all_results = {}

for m_idx, m_num in enumerate( np.arange(n_models).astype(int) ):
    
    # build a file name...
    if weighted_loss == 0:
        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob_train * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
    else:
        # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
        if stim_prob_train == 1 / n_afc:
            fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob_train * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'
        else:
            fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob_train * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'  

    
    # load the trained model, set to eval, requires_grad == False
    if device == 'cpu':
        net = load_model( fn,device )
    elif device == 'gpu':
        net = load_model_cuda( fn, args.gpu)
    elif device == 'mps':
        net = load_model_mps( fn )
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
    if rnn_settings['step'] >= iters:
        print(f"Skipping model {m_num} — steps = {rnn_settings['steps']}")
        continue
    else:
        
        # linear spacing first
        #scalars = [0, 0.1, 0.25, 0.5, 0.75, 1]
        # logarithmic spacing
        #scalars = [0, 0.1, 0.3, 0.6, 1] # a
        #scalars = [0, 0.01, 0.1, 0.2, 0.3, 1] #b
        # custom spacing based on prior observation
        scalars = [0, 0.1,0.15,0.2,0.3,0.5,0.7,1.0]
        
        inputs,cues,s_label,c_label = task.generate_rdk_reproduction_cue_stim()
        
        with torch.no_grad():
            res = measure_feedback_contributions(net, inputs, cues, scalars, rnn_settings)
        all_results[f'model_{m_idx}'] = res
    
    
df = pd.DataFrame()

for alpha in scalars:
    ratios = [all_results[m][alpha]['layer1'] for m in all_results]  # layer1 example
    df.loc[alpha, 'layer1_mean'] = np.mean(ratios)
    df.loc[alpha, 'layer1_std'] = np.std(ratios)

    ratios2 = [all_results[m][alpha]['layer2'] for m in all_results]
    df.loc[alpha, 'layer2_mean'] = np.mean(ratios2)
    df.loc[alpha, 'layer2_std'] = np.std(ratios2)


plt.figure(figsize=(7,5))

# Layer1
plt.plot(df.index, df['layer1_mean'], marker='o', color='blue', label='Layer1 mean')
plt.fill_between(df.index,
                 df['layer1_mean'] - df['layer1_std'],
                 df['layer1_mean'] + df['layer1_std'],
                 color='blue', alpha=0.2)

# Layer2
plt.plot(df.index, df['layer2_mean'], marker='s', color='orange', label='Layer2 mean')
plt.fill_between(df.index,
                 df['layer2_mean'] - df['layer2_std'],
                 df['layer2_mean'] + df['layer2_std'],
                 color='orange', alpha=0.2)

plt.xlabel('Feedback scalar (alpha)')
plt.ylabel('Feedback contribution fraction')
plt.title('Feedback Contributions Across Layers')
plt.legend()
plt.grid(True)
plt.show()


#normed
# For Layer1
max_layer1 = df['layer1_mean'].max()
df['layer1_mean_norm'] = df['layer1_mean'] / max_layer1
df['layer1_std_norm'] = df['layer1_std'] / max_layer1

# For Layer2
max_layer2 = df['layer2_mean'].max()
df['layer2_mean_norm'] = df['layer2_mean'] / max_layer2
df['layer2_std_norm'] = df['layer2_std'] / max_layer2

plt.figure(figsize=(7,5))

# Layer1
plt.plot(df.index, df['layer1_mean_norm'], marker='o', color='blue', label='Layer1')
plt.fill_between(df.index,
                 df['layer1_mean_norm'] - df['layer1_std_norm'],
                 df['layer1_mean_norm'] + df['layer1_std_norm'],
                 color='blue', alpha=0.2)

# Layer2
plt.plot(df.index, df['layer2_mean_norm'], marker='s', color='orange', label='Layer2')
plt.fill_between(df.index,
                 df['layer2_mean_norm'] - df['layer2_std_norm'],
                 df['layer2_mean_norm'] + df['layer2_std_norm'],
                 color='orange', alpha=0.2)

plt.xlabel('Feedback scalar (alpha)')
plt.ylabel('Normalized feedback contribution')
plt.title('Normalized Feedback Contributions Across Layers')
plt.legend()
plt.grid(True)
plt.show()
    
    