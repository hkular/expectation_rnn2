#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:41:19 2025

@author: hkular
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

# example cmd line...
# python decode_multilayer.py --gpu 0 --device gpu --classes stim --time_or_xgen 0 --cue_on 0 --cue_layer 3 --stim_prob_train 70 --stim_prob_eval 70 ...
# -- stim_noise_train 0.1 --stim_noise_eval 0.1 --stim_amp_train 1.0 --stim_amp_eval 1.0 --int_noise_train 0.1 --int_noise_eval 0.1

#--------------------------
# setup argparser
# #--------------------------
parser = argparse.ArgumentParser(description='Analyze RNNs')
# decoding params
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
parser.add_argument('--cue_on', required=True, type=int,
        help="When does cue come on?")
parser.add_argument('--cue_layer', required=True, type=int,
        help="Which layer receives the cue?")
parser.add_argument('--n_afc', required=False,type=int,
        default=6,help="How many stimulus alternatives?")
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
parser.add_argument('--batch_size', required=False,type=int, default = 2000,
        help="How many trials to eval?")
parser.add_argument('--weighted_loss', required=False,type=int, default = 0,
        help="Weighted loss (1) or unweighted loss(0)?")



args = parser.parse_args()

# for debugging
# args.task_type = 'rdk_repro_cue'
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

if classes == 'cue':
    n_stim_chans = num_cues
else:
    n_stim_chans = n_Afc

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
n_trials_per_batch = 256            # how many trials per eval batch?
num_targ_trials = 128               # keep evaluating batches till we have this many cor and incor trials for each stim type
max_batches = 500                   # max number of batches to eval trying to get enough cor/incor trials (some models are 100% all the time, so need an escape) 
n_t_steps = T // w_size
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
# matrices to store model decode acc
#--------------------------

# to store indices of exc and inh units and taus...
params_dict = {}
# storage arrays for model eval acc and decoding acc...
eval_acc = np.full( len(n_models),np.nan )
inp_stims = np.full( (len(n_models),h_size,n_stim_chans*2 ),np.nan )
h1_weights = np.full( (len(n_models),h_size,h_size),np.nan )
h2_weights = np.full( (len(n_models),h_size,h_size),np.nan )
h3_weights = np.full( (len(n_models),h_size,h_size),np.nan )
ff12_weights = np.full( (len(n_models),h_size,h_size),np.nan )
ff23_weights = np.full( (len(n_models),h_size,h_size),np.nan )
fb21_weights = np.full( (len(n_models),h_size,h_size),np.nan )
fb32_weights = np.full( (len(n_models),h_size,h_size),np.nan )
stim_id_cor = np.full( (len(n_models),num_targ_trials*n_stim_chans),np.nan )
all_stims_cor = np.full( (len(n_models),num_targ_trials*n_stim_chans,n_stim_chans*2),np.nan )
stim_id_incor = np.full( (len(n_models),num_targ_trials*n_stim_chans),np.nan )
all_stims_incor = np.full( (len(n_models),num_targ_trials*n_stim_chans,n_stim_chans*2),np.nan )
                       
if time_or_xgen == 0:
    cor_acc_exc1 = np.full( ( len(n_models),n_t_steps ),np.nan )
    cor_acc_inh1 = np.full( ( len(n_models),n_t_steps ),np.nan )
    cor_acc_exc2 = np.full( ( len(n_models),n_t_steps ),np.nan )
    cor_acc_inh2 = np.full( ( len(n_models),n_t_steps ),np.nan )
    cor_acc_exc3 = np.full( ( len(n_models),n_t_steps ),np.nan )
    cor_acc_inh3 = np.full( ( len(n_models),n_t_steps ),np.nan )                                    
    
    incor_acc_exc1 = np.full( ( len(n_models),n_t_steps ),np.nan )
    incor_acc_inh1 = np.full( ( len(n_models),n_t_steps ),np.nan )
    incor_acc_exc2 = np.full( ( len(n_models),n_t_steps ),np.nan )
    incor_acc_inh2 = np.full( ( len(n_models),n_t_steps ),np.nan ) 
    incor_acc_exc3 = np.full( ( len(n_models),n_t_steps ),np.nan )
    incor_acc_inh3 = np.full( ( len(n_models),n_t_steps ),np.nan )     
    
elif time_or_xgen == 1:
    cor_acc_exc1 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    cor_acc_inh1 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    cor_acc_exc2 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    cor_acc_inh2 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    cor_acc_exc3 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    cor_acc_inh3 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )                                    
    
    incor_acc_exc1 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    incor_acc_inh1 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    incor_acc_exc2 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    incor_acc_inh2 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan ) 
    incor_acc_exc3 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan )
    incor_acc_inh3 = np.full( ( len(n_models),n_t_steps,n_t_steps ),np.nan ) 

# alloc storage for average synaptic currents
cor_avg_h1 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
cor_avg_h2 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
cor_avg_h3 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
cor_avg_ff12 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
cor_avg_ff23 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
cor_avg_fb21 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
cor_avg_fb32 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )

incor_avg_h1 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
incor_avg_h2 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
incor_avg_h3 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
incor_avg_ff12 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
incor_avg_ff23 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
incor_avg_fb21 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )
incor_avg_fb32 = np.full( ( len(n_models),n_stim_chans,T,h_size),np.nan )

# fn out for npz file to store decoding data
if time_or_xgen == 0:
    fn_out = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob_train * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse.npz'
else:
    fn_out = f'decode_data/{task_type}_xgen_{classes}_{n_afc}nafc_stim_prob{int(stim_prob_train * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse.npz'

 # check to see if we already have this file...
if os.path.exists( fn_out ) == False:
     
 # to keep track of any model where we can't find 
 # enough cor and incor trials...
    over_max_batch = []
    for m_idx, m_num in enumerate( np.arange(n_models).astype(int) ):
        
        # build a file name...
        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob_train * 100 )}_stim_amp-{int( stim_amp_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
    
        # init dicts to store exc,inh indicies
        params_dict[ m_idx ] = {}
        
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
            sr_scram = np.zeros(1)
        print(f'loaded model {m_num}')
        if rnn_settings['step'] >= iters:
            print(f"Skipping model {m_num} — steps = {rnn_settings['steps']}")
            continue
        else:
        
            # update eval params
            net.recurrent_layer.batch_size = n_trials_per_batch
            net.recurrent_layer.noise = int_noise_eval
            
            # store the input weights (stims)
            inp_stims[ m_idx,:,: ] = net.recurrent_layer.inp_layer.weight.cpu().detach().numpy()
         
            # eval a batch of trials using the trained model
            sp_cor,sp_incor,cl,h1_cor,h1_incor,h2_cor,h2_incor,h3_cor,h3_incor,ff_cor,ff_incor,fb_cor,fb_incor,w1,w2,w3,ff12_w,ff23_w,fb21_w,fb32_w,exc1,exc2,exc3,inh1,inh2,inh3,tau1,tau2,tau3,eval_acc[ m_idx ],enough_cor,enough_incor = eval_model_batch( net, task, sr_scram, internal_noise, n_afc, num_targ_trials, max_batches )
            
 
            # first make sure that max batches wasn't exceeded even for cor trials (if so, don't contine to analyze this model)
            if enough_cor == True:
                # store weights
                h1_weights[ m_idx,:,: ] = w1
                h2_weights[ m_idx,:,: ] = w2
                h3_weights[ m_idx,:,: ] = w3
                ff12_weights[ m_idx,:,: ] = ff12_w
                ff23_weights[ m_idx,:,: ] = ff23_w
                fb21_weights[ m_idx,:,: ] = fb21_w
                fb32_weights[ m_idx,:,: ] = fb32_w
       
                # store indices
                params_dict[ m_idx ]['exc1'] = exc1
                params_dict[ m_idx ]['exc2'] = exc2
                params_dict[ m_idx ]['exc3'] = exc3
                params_dict[ m_idx ]['inh1'] = inh1
                params_dict[ m_idx ]['inh2'] = inh2
                params_dict[ m_idx ]['inh3'] = inh3
                params_dict[ m_idx ]['tau1'] = tau1
                params_dict[ m_idx ]['tau2'] = tau2
                params_dict[ m_idx ]['tau3'] = tau3
        
                # get stim presented on each trial
                stim_pres_cor = sp_cor[ :,:n_stim_chans ].nonzero()[1]
                #dis_pres_cor = sp_cor[ :,n_stim_chans: ].nonzero()[1] use this to store cue instead?
                if enough_incor == True:
                    stim_pres_incor = sp_incor[ :,:n_stim_chans ].nonzero()[1]
                    #dis_pres_incor = sp_incor[ :,n_stim_chans: ].nonzero()[1] use this to store cue instead?
                
                # store them for this model to save out
                stim_id_cor[ m_idx,: ] = stim_pres_cor
                all_stims_cor[ m_idx,:,: ] = sp_cor
                if enough_incor == True:
                    stim_id_incor[ m_idx,: ] = stim_pres_incor
                    all_stims_incor[ m_idx,:,: ] = sp_incor

                # loop over stims to compute mean across all like trials
                for s in range ( n_stim_chans ):
                    
                    cor_ind = ( stim_pres_cor==s )
                    if np.sum( cor_ind )>0:
                        cor_avg_h1[ m_idx,s,:,: ] = np.mean( h1_cor[ :,cor_ind,: ],axis=1 ) 
                        cor_avg_h2[ m_idx,s,:,: ] = np.mean( h2_cor[ :,cor_ind,: ],axis=1 )
                        cor_avg_h3[ m_idx,s,:,: ] = np.mean( h3_cor[ :,cor_ind,: ],axis=1 ) 
                        cor_avg_ff12[ m_idx,s,:,: ] = np.mean( ff12_cor[ :,cor_ind,: ],axis=1 )
                        cor_avg_ff23[ m_idx,s,:,: ] = np.mean( ff23_cor[ :,cor_ind,: ],axis=1 )
                        cor_avg_fb21[ m_idx,s,:,: ] = np.mean( fb_cor21[ :,cor_ind,: ],axis=1 )
                        cor_avg_fb32[ m_idx,s,:,: ] = np.mean( fb_cor32[ :,cor_ind,: ],axis=1 )

                    if enough_incor == True:
                        incor_ind = ( stim_pres_incor==s )
                        if np.sum( incor_ind )>0:
                            incor_avg_h1[ m_idx,s,:,: ] = np.mean( h1_incor[ :,incor_ind,: ],axis=1 ) 
                            incor_avg_h2[ m_idx,s,:,: ] = np.mean( h2_incor[ :,incor_ind,: ],axis=1 )
                            incor_avg_h3[ m_idx,s,:,: ] = np.mean( h3_incor[ :,incor_ind,: ],axis=1 ) 
                            incor_avg_ff12[ m_idx,s,:,: ] = np.mean( ff12_incor[ :,incor_ind,: ],axis=1 )
                            incor_avg_ff23[ m_idx,s,:,: ] = np.mean( ff23_incor[ :,incor_ind,: ],axis=1 )
                            incor_avg_fb21[ m_idx,s,:,: ] = np.mean( fb21_incor[ :,incor_ind,: ],axis=1 )
                            incor_avg_fb32[ m_idx,s,:,: ] = np.mean( fb32_incor[ :,incor_ind,: ],axis=1 )

                # -------------------------------------
                # do the cross-validated decoding for correct trials       
                # -------------------------------------
                print(f'Decoding correct trials')
                cor_acc_exc1[ m_idx,: ],cor_acc_inh1[ m_idx,: ],cor_acc_exc2[ m_idx,: ],cor_acc_inh2[ m_idx,: ] = decode_ls_svm(h1_cor, h2_cor, n_stim_chans, stim_pres_cor, t_step, exc1, inh1, exc2, inh2, train_prcnt, n_cvs, tmpnt_or_cross_tmpnt, num_targ_trials)

                if enough_incor == True:
                    print(f'Decoding incorrect trials')
                    incor_acc_exc1[ m_idx,: ],incor_acc_inh1[ m_idx,: ],incor_acc_exc2[ m_idx,: ],incor_acc_inh2[ m_idx,: ] = decode_ls_svm(h1_incor, h2_incor, n_stim_chans, stim_pres_incor, t_step, exc1, inh1, exc2, inh2, train_prcnt, n_cvs, tmpnt_or_cross_tmpnt, num_targ_trials)
            
            # if max_batches exceeded, record which model and then move on
            else:
                over_max_batch.append( m_idx )
                
        # save out the data across models
        np.savez( fn_out,n_tmpts=T,over_max_batch=over_max_batch,
                 cor_acc_exc1=cor_acc_exc1,cor_acc_inh1=cor_acc_inh1,
                 cor_acc_exc2=cor_acc_exc2,cor_acc_inh2=cor_acc_inh2,
                 cor_acc_exc3=cor_acc_exc3,cor_acc_inh3=cor_acc_inh3,
                 incor_acc_exc1=incor_acc_exc1,incor_acc_inh1=incor_acc_inh1,
                 incor_acc_exc2=incor_acc_exc2,incor_acc_inh2=incor_acc_inh2,
                 incor_acc_exc3=incor_acc_exc3,incor_acc_inh3=incor_acc_inh3,
                 eval_acc=eval_acc,h1_weights=h1_weights,h2_weights=h2_weights,
                 h3_weights=h3_weights,ff12_weights=ff12_weights,
                 ff23_weights=ff23_weights,fb21_weights=fb21_weights,
                 fb32_weights=fb32_weights,
                 stim_id_cor=stim_id_cor,all_stims_cor=all_stims_cor,
                 stim_id_incor=stim_id_incor,all_stims_incor=all_stims_incor,
                 cor_avg_h1=cor_avg_h1,cor_avg_h2=cor_avg_h2,cor_avg_h3=cor_avg_h3,
                 cor_avg_ff12=cor_avg_ff12,cor_avg_ff23=cor_avg_ff23,
                 cor_avg_fb21=cor_avg_fb21,cor_avg_fb32=cor_avg_fb32,
                 incor_avg_h1=incor_avg_h1,incor_avg_h2=incor_avg_h2,
                 incor_avg_h3=incor_avg_h3,incor_avg_ff12=incor_avg_ff12,
                 incor_avg_ff23=incor_avg_ff23,incor_avg_fb21=incor_avg_fb21,
                 incor_avg_fb32=incor_avg_fb32,params_dict=params_dict )
                
           

