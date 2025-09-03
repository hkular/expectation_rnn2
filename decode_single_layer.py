#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:54:05 2025

@author: johnserences and hkular

TODO
- currently you have to manually change which layer you're decoding in lines 180, 188, 193
- currently you have to manually change which decoding you're plotting in lines 209, 210

"""

#--------------------------
# imports
#--------------------------
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt   
from scipy.stats import sem
from helper_funcs import *
from rdk_task import RDKtask
import matplotlib.colors
from model_count import count_models
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# set up some nice colors (from Robert and Nuttida)
hex_c = ['#06D2AC',  # aqua green
 '#206975',  # deep teal
 '#6F3AA4',  # rich purple
 '#2B1644',  # deep plum
 '#529CA0',  # soft blue-teal
 '#A05DC0']  # lavender violet

#--------------------------
# Basic model params
#--------------------------
device = 'cpu'                    # device to use for loading/eval model
task_type = 'rdk_repro_cue'       # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                         # number of stimulus alternatives
T = 210                          # timesteps in each trial
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_prob_train = 0.7                  # during training probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
stim_prob_eval = 1/n_afc     # eval the model at this prob level
stim_amp_train = 1.0             # stim amplitude during training
stim_amp_eval = 1.0                   # stim amplitude during eval
stim_noise_train = 0.1           # external stim noise during training  
stim_noise_eval = 0.8     # magnitude of randn background noise in the stim channel for eval
int_noise_train = 0.1
int_noise_eval = 0.1
batch_size = 2000                 # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct
weighted_loss = 0                 #  0 = nw_mse l2 or 1 = weighted mse
noise_internal = 0.1              # trained under 0.1 try 0.25 
num_cues = 2                      # how many s->r cues
cue_on = 0        # cue comes on after stim goes off
cue_dur = T-cue_on                # cue stays on until the end
plots = True
cue_layer = 3
rand_seed_bool = True
if task_type == 'rdk':
    fn_stem = 'trained_models_rdk/gonogo_'
    out_size = 1
elif task_type == 'rdk_reproduction':
    fn_stem = 'trained_models_rdk_reproduction/repro_'
    out_size = n_afc  
elif task_type == 'rdk_repro_cue':
    fn_stem = f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer}/reprocue_'
    out_size = n_afc  
  
fb21_scalar = 1.0
fb32_scalar = 1.0
#--------------------------
# decoding params
#--------------------------
trn_prcnt = 0.8    # percent of data to use for training
n_cvs = 5  # how many train/test cv folds
time_or_xgen = 0   # decode timepnt x timepnt or do full xgen matrix
classes = 'stim'   # which class? stim, choice
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
            'stim_prob' : stim_prob_eval, 'stim_amp' : stim_amp_eval, 'stim_noise' : stim_noise_eval, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur, 'rand_seed_bool':rand_seed_bool}

# create the task object
task = RDKtask( settings )


#--------------------------
# How many trained models in this cond
#--------------------------
n_models = 1#count_models(n_afc, stim_prob, stim_amps_train, stim_noise_train, weighted_loss, task_type, fn_stem, directory = os.getcwd())

# fn out for npz file to store decoding data
#fn_out = f'{fn_stem}_decode_n_afc-{n_afc}_stim_noise-{stim_noise_train}_trnamp-{da}_evalamp-{ea}_distype-{stim_dis_weights}_ff-{ffa}_fb-{fba}_inh_kappa-{ring_w_kappa_inh}_cue_on-{cue_on}_h1_L2-{h1_lam_l2}_h2_L2-{h2_lam_l2}_ST-{sparse_type}_fb_on-{fb_on}_stim_noise_eval-{stim_noise_eval}_internal_noise-{internal_noise_eval}.npz'


#--------------------------
# matrices to store model decode acc
#--------------------------

if time_or_xgen == 0:
    over_acc = np.full( ( n_models,T//w_size ),np.nan )
    stim_acc = np.full( ( n_models,n_afc,T//w_size ),np.nan )
else:
    over_acc = np.full( ( n_models,T//w_size,T//w_size ),np.nan )
    stim_acc = np.full( ( n_models,n_afc,T//w_size,T//w_size ),np.nan )    
      

for m_num in np.arange( n_models ):
    
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
    if device == 'cuda':
        net = load_model( fn, cuda_num )
    elif device == 'cpu':   
        net = load_model( fn,device )
    print(f'loaded model {m_num}')
    
    # load cue scramble matrix
    if task_type == 'rdk_repro_cue':
        with open(f'{fn[:-3]}.json', "r") as infile:
           _ , rnn_settings = json.load(infile)
        sr_scram_list = rnn_settings['sr_scram']
        sr_scram_list = [sr_scram_list[str(s)] for s in sorted(sr_scram_list.keys(), key=int)]
        sr_scram = np.array(sr_scram_list)
    else:
        sr_scram = []
    
    # update internal noise
    net.recurrent_layer.noise = int_noise_eval
    # update fb21 and fb32
    with torch.no_grad():
        net.recurrent_layer.h_layer2.wfb_21.mul_(fb21_scalar)
        net.recurrent_layer.h_layer3.wfb_32.mul_(fb32_scalar)

    # eval a batch of trials using the trained model
    outputs,s_label,w1,w2,w3,exc1,inh1,exc2,inh2,exc3,inh3,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3,m_acc,tbt_acc,cues = eval_model( net, task, sr_scram )
    s_label_int = np.argmax(s_label, axis=1)
    
    if plots:
        # quick plots - first expected, correct and incorrect trials
        plt.plot(np.squeeze( outputs[ :,(s_label_int==0) & (tbt_acc==1),0 ] ), c=hex_c[0], alpha=0.25 )
        plt.plot(np.squeeze( outputs[ :,(s_label_int==0) & (tbt_acc==0),0 ] ), c=hex_c[1], alpha=0.25 )
        plt.xlabel('Time Step')
        plt.ylabel('Output')
        plt.title(f'expected {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss mod-{m_num}')
        plt.show()
        
        # quick plots - the unexpected, correct and incorrect trials
        blues = plt.cm.Blues(np.linspace(0.4, 0.9, out_size))  # Skip very light colors
        reds = plt.cm.Reds(np.linspace(0.4, 0.9, out_size))
        # Convert RGBA to hex
        hex_c_correct = [matplotlib.colors.rgb2hex(c[:3]) for c in blues]
        hex_c_incorrect = [matplotlib.colors.rgb2hex(c[:3]) for c in reds]
        for i in np.arange(out_size):
            plt.plot(np.squeeze( outputs[ :,(s_label_int!=0) & (tbt_acc==1),i ] ), c=hex_c_correct[i], alpha=0.25 )
            plt.plot(np.squeeze( outputs[ :,(s_label_int!=0) & (tbt_acc==0),i ] ), c=hex_c_incorrect[i], alpha=0.25 )
            plt.xlabel('Time Step')
            plt.ylabel('Output')
        #plt.ylim((-0.2,1.1))
        plt.title(f'unexpected {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss mod-{m_num}')
        plt.show()
    
    if time_or_xgen == 0:
        tmp_over_acc = np.full((n_cvs,T//w_size),np.nan)
        tmp_stim_acc = np.full((n_cvs,n_afc,T//w_size),np.nan)
    else:
        tmp_over_acc = np.full( ( n_cvs,T//w_size,T//w_size ),np.nan )
        tmp_stim_acc = np.full( ( n_cvs,n_afc,T//w_size,T//w_size ),np.nan )    
        
    #--------------------------
    # decoding:
    #--------------------------
    for cv in range( n_cvs ):
        if classes == 'stim':
            tmp_over_acc[cv,:], tmp_stim_acc[cv,:,:] = decode_ls_svm(h3, s_label_int, n_afc, w_size, time_or_xgen, trn_prcnt)          
        elif classes == 'choice':
            # Step 1: Average output over the response window
            mean_outputs = np.mean(outputs[stim_on:, :, :], axis=0)  # shape: (n_trials, n_stims)
            # Step 2: Decode the choice (argmax across stimulus channels)
            choice_labels = np.argmax(mean_outputs, axis=1)  # shape: (n_trials,)
            tmp_over_acc[cv,:], tmp_stim_acc[cv,:,:] = decode_ls_svm(h3, s_label_int, n_afc, w_size, time_or_xgen, trn_prcnt)
    # average over cvs
    over_acc[m_num,:] = np.mean(tmp_over_acc,axis=0)
    stim_acc[m_num,:,:] = np.mean(tmp_stim_acc,axis=0)
    print(f"done decoding {classes}")
#--------------------------
# plot decoding stuff...figure out if time x time 
# or xgen, and if time x time if more than one model 
# in which case compute sem.
# manually change which decoding you're plotting 
#--------------------------
if time_or_xgen == 0: 
    
    
    
    # mean decoding and sem over models   
    m_over_acc = np.mean(over_acc,axis=0)
    m_stim_acc = np.mean(stim_acc,axis=0)
    if m_num > 0: 
        
        print(f'plotting mean and sem over models') 
        
        sem_over_acc = sem(over_acc,axis=0)
        sem_stim_acc = sem(stim_acc,axis=0)
    
        # plots...
        t = np.arange(0,T,w_size)
        plt.errorbar( t, m_over_acc, sem_over_acc, fmt=hex_c[0], label = 'all stims' )
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title(f' {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss')
        plt.show()
        
        # expected vs mean of the rest
        plt.errorbar( t, m_stim_acc[0,:],sem_stim_acc[0,:], fmt=hex_c[0], label = 'expected' )
        plt.errorbar( t, np.mean( m_stim_acc[1:,:],axis=0 ), np.mean( sem_stim_acc[1:,:],axis=0 ), fmt=hex_c[1], label = 'unexpected' )
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title(f'expected vs unexpected {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss')
        plt.legend()
        plt.show()
        
        # plot all stims
        plt.errorbar( t, m_stim_acc[0,:],sem_stim_acc[0,:], fmt=hex_c[0], label = 'expected' )
        for i in range(n_afc-1):
            plt.errorbar( t, m_stim_acc[i+1,:], sem_stim_acc[i+1,:], fmt=hex_c[i+1], label = f'unexpected {i+1}' )
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title(f'expected vs unexpected {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss')
        plt.legend()
        #plt.savefig(f'plots/decoding/{fn[15:-4]}0-{n_models -1}.png')   
        plt.show()
        
        
    else:
        
        print(f'plotting only one model')
        
        
        # plots...
        t = np.arange(0,T,w_size)
        plt.plot( t, m_over_acc, c=hex_c[0] )
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title(f'{n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss')
        plt.show()
        
        # expected vs mean of the rest
        plt.plot( t, m_stim_acc[0,:], c=hex_c[0], label = 'expected' )
        plt.plot( t, np.mean( m_stim_acc[1:,:],axis=0 ), c=hex_c[1], label = 'unexpected' )
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title(f' {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss')
        plt.legend()
        plt.show()

        # all stims decoding accuracy individually
        plt.plot( t, m_stim_acc[0,:], c=hex_c[0], label = 'expected' )
        for i in range(n_afc-1):
            plt.plot( t, m_stim_acc[i+1,:], c=hex_c[i+1], label = f'unexpected {i+1}' )
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title(f' {n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss')
        plt.legend()
        plt.show()       

# plot xgen
else:
    
    print('plotting cross generalization')
    
    #  mean decoding over models
    m_over_acc = np.mean(over_acc,axis=0)
    m_stim_acc = np.mean(stim_acc,axis=0)
    
    # overall acc
    plt.imshow(m_over_acc,origin='lower',aspect='equal')
    plt.colorbar()
    plt.title(f'{n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss overall')
    plt.show()
    
    # for each stim - first expected
    plt.imshow(m_stim_acc[0,:,:],origin='lower',aspect='equal')
    plt.colorbar()
    plt.title(f'{n_afc}-afc {stim_amp_train}s-amp {weighted_loss}-weighted_loss expected')
    plt.show()    
    
    # for each stim - then avg of unexpected
    plt.imshow(np.mean(m_stim_acc[1:,:,:],axis=0),origin='lower',aspect='equal')
    plt.colorbar()
    plt.title(f'{n_afc}-afc {stim_amp_train}-amp {weighted_loss}-weighted_loss unexpected')
    plt.show()        
    
# save data if it's complete
# if n_models == 10:
#     # save out the data across models
#     np.savez( fn_out,n_tmpts=T,pred_acc_exc1=pred_acc_exc1,pred_acc_inh1=pred_acc_inh1,
#              pred_acc_exc2=pred_acc_exc2,pred_acc_inh2=pred_acc_inh2,eval_acc=eval_acc,
#              stim_id=stim_id,all_stims=all_stims,inp_stims=inp_stims,avg_h1=avg_h1,avg_h2=avg_h2,
#              avg_ff=avg_ff,avg_fb=avg_fb,params_dict=params_dict )
    
# at the end remind me which one we were working on  
print(f'finished {settings}')
print('\007') # make a sound   