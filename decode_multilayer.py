#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:00:15 2025

@author: hkular
"""
#--------------------------
# imports
#--------------------------
import os
os.environ["OMP_NUM_THREADS"] = "4" #try limiting threads because of multithreading overhead and greedy BLAS
import numpy as np
import json
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from scipy.stats import sem
from helper_funcs import *
from rdk_task import RDKtask
from numpy import trapezoid
from model_count import count_models
import argparse

# example cmd line...
# python decode_multilayer.py --gpu 0 --device gpu --classes stim --time_or_xgen 0 --cue_on 0 --cue_layer 3 --stim_prob_train 70 --stim_prob_eval 70 ...
# -- stim_noise_train 0.1 --stim_noise_eval 0.1 --stim_amp_train 1.0 --stim_amp_eval 1.0 --int_noise_train 0.1 --int_noise_eval 0.1

#--------------------------
# setup argparser
# #--------------------------
parser = argparse.ArgumentParser(description='Analyze RNNs')
# # decoding params
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu?")
parser.add_argument('--device', required=False,
        default='cpu', help="gpu or cpu?")
parser.add_argument('--classes', required=False,
        default='stim', help="stim or choice or cue?")
parser.add_argument('--time_or_xgen', required=True,
                    type=int, help="time=0, xgen=1")
# model params
parser.add_argument('--task_type', required=False,type=str,
        default='rdk_repro_cue', help="Which task for train and eval?")
parser.add_argument('--T', required=False, type=int,
        default='210', help="How long is the trial?")
parser.add_argument('--cue_on', required=False, type=int,
        default = '0',help="When does cue come on?")
parser.add_argument('--cue_layer', required=False, type=int,
        default = '3',help="Which layer receives the cue?")
parser.add_argument('--n_afc', required=False,type=int,
        default='6',help="How many stimulus alternatives?")
parser.add_argument('--stim_prob_train', required=True,type=int,
        help="What stim prob during training?")
parser.add_argument('--stim_prob_eval', required=True,type=int,
        help="What stim prob during eval?")
parser.add_argument('--stim_noise_train', required=True,type=float,
        help="What stim noise during training?")
parser.add_argument('--stim_noise_eval', required=True,type=float,
        help="What stim noise during eval?")
parser.add_argument('--stim_amp_train', required=True,type=float,
        help="What stim amp during training?")
parser.add_argument('--stim_amp_eval', required=True,type=float,
        help="What stim amp during eval?")
parser.add_argument('--int_noise_train', required=False,type=float,
        default=0.1,help="What internal noise during training?")
parser.add_argument('--int_noise_eval', required=False,type=float,
        default=0.1, help="What internal noise during eval?")
parser.add_argument('--batch_size', required=False,type=int, default = '2400',
        help="How many trials to eval?")
parser.add_argument('--weighted_loss', required=False,type=int, default = '0',
        help="Weighted loss (1) or unweighted loss(0)?")
parser.add_argument('--fb21_scalar', required=False,type=float, default = '1.0',
        help="Feedback from layer 2 to 1 scalar?")
parser.add_argument('--fb32_scalar', required=False,type=float, default = '1.0',
        help="Feedback from layer 3 to 2 scalar?")


args = parser.parse_args()

# for debugging
# args.task_type = 'rdk_reproduction'
# args.gpu = 1
# args.device = 'cpu'
# args.classes = 'cue'
# args.time_or_xgen = 0
# args.n_afc = 6
# args.T = 210
# args.cue_on = 75
# args.cue_layer = 3
# args.stim_prob_train = 1 / args.n_afc
# args.stim_prob_eval = 1 / args.n_afc
# args.stim_noise_train = 0.1
# args.stim_noise_eval = 0.1
# args.stim_amp_train = 1.0
# args.stim_amp_eval = 1.0
# args.int_noise_train = 0.1
# args.int_noise_eval = 0.1
# args.batch_size = 2000
# args.weighted_loss = 0
# args.fb21_scalar = 1.0
# args.fb32_scalar = 1.0


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
    stim_prob_eval = args.stim_prob_eval /100     # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
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
fb21_scalar = args.fb21_scalar                      # feedback scaled by? trained with 1.0
fb32_scalar = args.fb32_scalar                      # feedback scaled by? trained with 1.0
num_cues = 2                                        # how many sr_scram
stim_on = 50                                        # timestep of stimulus onset
stim_dur = 25                                       # stim duration
cue_dur = T-cue_on                                  # on the rest of the trial
acc_amp_thresh = 0.8                                # to determine acc of model output: > acc_amp_thresh during target window is correct
h_size = 200                                        # how many units in a hidden layer
plots = False                                       # only plot if not run through terminal
iters = 200000                                      # cut off during training
rand_seed_bool = True                               # do we want eval trials to be reproducible
equal_balance = True                                # do we want the same number of trials per stim
if task_type == 'rdk':
    fn_stem = 'trained_models_rdk/gonogo_'
    out_size = 1
elif task_type == 'rdk_reproduction':
    fn_stem = f'trained_models_rdk_reproduction/timing_{T}/repro_'
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
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur, 'rand_seed_bool':rand_seed_bool}

# create the task object
task = RDKtask( settings )


#--------------------------
# How many trained models in this cond
#--------------------------
n_models = count_models(n_afc, stim_prob_train, stim_amp_train, stim_noise_train, weighted_loss, task_type, fn_stem, directory = os.getcwd())
n_layers = 3

#--------------------------
# matrices to store model decode acc
#--------------------------
m_acc = np.full((n_models), np.nan)
outputs = np.full((n_models,T, batch_size, n_afc), np.nan)
s_labels = np.full((n_models,T, batch_size), np.nan)
tbt_acc = np.full((n_models,batch_size), np.nan)
cues_int = np.full((n_models, batch_size), np.nan)
if time_or_xgen == 0:
    over_acc = np.full( ( n_models,n_layers,T//w_size ),np.nan )
    stim_acc = np.full( ( n_models,n_layers,n_afc,T//w_size ),np.nan )
else:
    over_acc = np.full( ( n_models,n_layers,T//w_size,T//w_size ),np.nan )
    stim_acc = np.full( ( n_models,n_layers,n_afc,T//w_size,T//w_size ),np.nan )
        
        
      
# to store indices of exc and inh units and taus...
#params_dict = {}

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
    
    # fn out for npz file to store decoding data
    if task_type == 'rdk_reproduction':
        if time_or_xgen == 0:
            fn_out = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob_train * 100)}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
        else:
            fn_out = f'decode_data/{task_type}_xgen_{classes}_{n_afc}nafc_stim_prob{int(stim_prob_train * 100)}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'

    elif task_type == 'rdk_repro_cue':
        if time_or_xgen == 0:
            fn_out = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob_train * 100)}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
        else:
            fn_out = f'decode_data/{task_type}_xgen_{classes}_{n_afc}nafc_stim_prob{int(stim_prob_train * 100)}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'

    # init dicts to store exc,inh indicies
    #params_dict[ m_idx ] = {}
    
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
        with open(f'{fn[:-3]}.json', "r") as infile:
           _ , rnn_settings = json.load(infile)
        sr_scram = []
    print(f'loaded model {m_num}')
    if rnn_settings['step'] >= iters:
        print(f"Skipping model {m_num} — steps = {rnn_settings['steps']}")
        continue
    else:
    
    
        
        # update internal noise
        net.recurrent_layer.noise = int_noise_eval
        # update fb21 and fb32
        with torch.no_grad():
            net.recurrent_layer.h_layer2.wfb_21.mul_(fb21_scalar)
            net.recurrent_layer.h_layer3.wfb_32.mul_(fb32_scalar)
    
        # eval a batch of trials using the trained model
        outputs[m_idx,:],s_label,h1,h2,h3,m_acc[m_idx],tbt_acc[m_idx,:],cues = eval_model_light( net, task, sr_scram, equal_balance )
        # s_label is a diff shape for cue version, deal with if statement later
        if task_type == 'rdk_repro_cue':
            s_label_int = np.argmax(s_label, axis=1)
            s_labels[m_idx,:] = s_label_int
            cues_int[m_idx,:] = np.argmax(cues[100,:,:].cpu().detach().numpy(), axis =1) # at t=100 cue should be on in all conditions change if we do early cue offset
        else:
            s_label_int = s_label.astype(int)
        # store indices
        # params_dict[ m_idx ]['exc1'] = exc1
        # params_dict[ m_idx ]['exc2'] = exc2
        # params_dict[ m_idx ]['exc3'] = exc3
        # params_dict[ m_idx ]['inh1'] = inh1
        # params_dict[ m_idx ]['inh2'] = inh2
        # params_dict[ m_idx ]['inh3'] = inh3
        # params_dict[ m_idx ]['tau1'] = tau1
        # params_dict[ m_idx ]['tau2'] = tau2
        # params_dict[ m_idx ]['tau3'] = tau3
    
        #  decoding
        layer_data = [h1, h2, h3]
        for l_num in np.arange(n_layers):
            
            if time_or_xgen == 0:
                tmp_over_acc = np.full((n_cvs,T//w_size),np.nan)
                tmp_stim_acc = np.full((n_cvs,n_afc,T//w_size),np.nan)
            else:
                tmp_over_acc = np.full((n_cvs,T//w_size,T//w_size),np.nan)
                tmp_stim_acc = np.full((n_cvs,n_afc,T//w_size,T//w_size),np.nan)
            
            
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
       

# save out the data across models
np.savez( fn_out,n_tmpts=T,m_acc=m_acc,
         over_acc=over_acc,stim_acc=stim_acc,stim_label=s_labels,outputs=outputs,
         cues=cues_int,tbt_acc=tbt_acc
          ) # params_dict=params_dict h1=h1,h2=h2,h3=h3, exclude for now too big
 
if plots:
    
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
    
        fig.suptitle(f'Decoding Accuracy {n_afc}-n_afc {int(stim_prob_train*100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)
    
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
    
        fig.suptitle(f'Decoding xgen {n_afc}-n_afc {int(stim_prob_eval*100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)
    
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
    
        fig.suptitle(f'Decoding xgen expected {n_afc}-n_afc {int(stim_prob*_eval100)}-stim_prob cue_on{cue_on} n_models{m_num+1}', fontsize=16)
    
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
    print('\007') # make a sound   
print(f'finished {settings}')