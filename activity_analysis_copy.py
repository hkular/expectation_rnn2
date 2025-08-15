#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:27:18 2025

@author: johnserences
"""

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from helper_funcs import *
from numpy import pi
from scipy.stats import circmean
import json 
import os
import multiprocessing
import argparse

# set up argparser
# python activity_analysis.py --gpu 0 --n_afc 2
parser = argparse.ArgumentParser(description='Analyze RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu?")
parser.add_argument('--n_afc', required=True,type=int,
        help="How many stimulus alternatives?")
args = parser.parse_args()
# args.gpu = 0
# args.n_afc = 12

# set up some nice colors (from Robert and Nuttida)
hex_c = ['#06D2AC', '#206975', '#6F3AA4', '#2B1644']

# conversions
d2r = pi/180
r2d = 180/pi
     
# -----------------------------
# set params for model eval
# -----------------------------
cuda_to_eval = args.gpu
m_nums = np.arange(10).astype(int)
task_type = 'repro_dis_cue'
stim_type = ['focal']
n_stim_chans = args.n_afc
stim_dis_weights = 'same'
dis_static_dynamic = 'dynamic'
num_cues = 2
cue_onsets = [0,180]
h1_lam_l2 = 0.0
L2s = [0.0]
sparse_type = 'weights'
fb_on = 0
ring_w_kappa_inh = 1.0
model_folder = 'trained_models'

dis_amps = [0,1]
eval_amps = [0,1]

# ff and fb amps to eval
ff_amps = [1]
fb_amps = [1]

train_fb = True
tmpnt_or_cross_tmpnt = 1     # do decoding timepoint by timepoint (0) or with cross-time generalization (1)
n_trials_per_batch = 256     # how many trials per eval batch?
num_targ_trials = 128        # keep evaluating batches till we have this many cor and incor trials for each stim type
max_batches = 500            # max number of batches to eval trying to get enough cor/incor trials (some models are 100% all the time, so need an escape) 

# timing/size stuff...
T = 240
r_size = 240
h_size = 480
stim_on = 20                      # timestep of stimulus onset (set above so both dicts)
stim_dur = 50                     # stim duration
t_step = 5
n_cvs = 2
train_prcnt = 0.5
internal_noise_eval = 0.1
stim_amp = 1.0
stim_noise_train = 0.0
stim_noise_levels_to_eval = [0.0,1.0]

#--------------------------------
# loop over stim types
#-------------------------------- 
for st in stim_type: 
    
    for cue_on in cue_onsets:
    
        for stim_noise_eval in stim_noise_levels_to_eval:
        
            #--------------------------------
            # loop over lambdas
            #--------------------------------        
            for h2_lam_l2 in L2s: 
                    
                # compute delay interval
                total_delay = int( ( T - stim_dur ) - ( stim_on + stim_dur ) )
                n_t_steps = T // t_step
                
                # loop over distractor amps
                for da_idx,da in enumerate( dis_amps ):
                    ea_idx = da_idx
                    ea = da
                    
                    # # loop over eval amps
                    # for ea_idx,ea in enumerate( eval_amps ):        
                    
                    # loop over fb levels (in descending order)
                    for ffa_idx,ffa in enumerate( ff_amps ):            
        
                        # loop over fb levels (in descending order)
                        for fba_idx,fba in enumerate( fb_amps ):
                
                            # storage arrays for model eval acc and decoding acc...
                            eval_acc = np.full( len(m_nums),np.nan )
                            inp_stims = np.full( (len(m_nums),r_size,n_stim_chans*2 ),np.nan )
                            rand_weights = np.full( (len(m_nums),h_size,h_size),np.nan )
                            ff_weights = np.full( (len(m_nums),r_size,h_size),np.nan )
                            fb_weights = np.full( (len(m_nums),h_size,r_size),np.nan )
                            stim_id_cor = np.full( (len(m_nums),num_targ_trials*n_stim_chans),np.nan )
                            all_stims_cor = np.full( (len(m_nums),num_targ_trials*n_stim_chans,n_stim_chans*2),np.nan )
                            stim_id_incor = np.full( (len(m_nums),num_targ_trials*n_stim_chans),np.nan )
                            all_stims_incor = np.full( (len(m_nums),num_targ_trials*n_stim_chans,n_stim_chans*2),np.nan )
                            
                            # to store indices of exc and inh units and taus...
                            params_dict = {}
                            
                            if tmpnt_or_cross_tmpnt == 0:
                                cor_acc_exc1 = np.full( ( len(m_nums),n_t_steps ),np.nan )
                                cor_acc_inh1 = np.full( ( len(m_nums),n_t_steps ),np.nan )
                                cor_acc_exc2 = np.full( ( len(m_nums),n_t_steps ),np.nan )
                                cor_acc_inh2 = np.full( ( len(m_nums),n_t_steps ),np.nan )                                    
                                
                                incor_acc_exc1 = np.full( ( len(m_nums),n_t_steps ),np.nan )
                                incor_acc_inh1 = np.full( ( len(m_nums),n_t_steps ),np.nan )
                                incor_acc_exc2 = np.full( ( len(m_nums),n_t_steps ),np.nan )
                                incor_acc_inh2 = np.full( ( len(m_nums),n_t_steps ),np.nan )      
                                
                            elif tmpnt_or_cross_tmpnt == 1:
                                cor_acc_exc1 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )
                                cor_acc_inh1 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )
                                cor_acc_exc2 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )
                                cor_acc_inh2 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )                                    
                                
                                incor_acc_exc1 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )
                                incor_acc_inh1 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )
                                incor_acc_exc2 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan )
                                incor_acc_inh2 = np.full( ( len(m_nums),n_t_steps,n_t_steps ),np.nan ) 
                            
                            # alloc storage for average synaptic currents
                            cor_avg_h1 = np.full( ( len(m_nums),n_stim_chans,T,r_size),np.nan )
                            cor_avg_h2 = np.full( ( len(m_nums),n_stim_chans,T,h_size),np.nan )
                            cor_avg_ff = np.full( ( len(m_nums),n_stim_chans,T,h_size),np.nan )
                            cor_avg_fb = np.full( ( len(m_nums),n_stim_chans,T,r_size),np.nan )

                            incor_avg_h1 = np.full( ( len(m_nums),n_stim_chans,T,r_size),np.nan )
                            incor_avg_h2 = np.full( ( len(m_nums),n_stim_chans,T,h_size),np.nan )
                            incor_avg_ff = np.full( ( len(m_nums),n_stim_chans,T,h_size),np.nan )
                            incor_avg_fb = np.full( ( len(m_nums),n_stim_chans,T,r_size),np.nan )
                            enough_cor_trials = np.full( len(m_nums),np.nan )
                            enough_incor_trials = np.full( len(m_nums),np.nan )
                            
                            # fn out for npz file to store decoding data
                            fn_out = f'decode_data/{task_type}_{st}_delay-{total_delay}_decode_nstims-{n_stim_chans}_stim_noise-{stim_noise_train}_trnamp-{da}_evalamp-{ea}_distype-{stim_dis_weights}_ff-{ffa}_fb-{fba}_inh_kappa-{ring_w_kappa_inh}_cue_on-{cue_on}_h1_L2-{h1_lam_l2}_h2_L2-{h2_lam_l2}_ST-{sparse_type}_fb_on-{fb_on}_stim_noise_eval-{stim_noise_eval}_internal_noise-{internal_noise_eval}.npz'
                                
                            # check to see if we already have this file...
                            if os.path.exists( fn_out ) == False:
                                
                                # loop over model instantiations
                                for m_idx,m in enumerate( m_nums ):
                                    
                                    # init dicts to store exc,inh indicies
                                    params_dict[ m_idx ] = {}
                                    
                                    # model name
                                    fn = f'trained_models/{task_type}_{st}_n_afc-{n_stim_chans}_stim_noise-{stim_noise_train}_disamp-{int( da * 100 )}_dis_type-{stim_dis_weights}_sd-{dis_static_dynamic}_n_cue-{num_cues}_cue_on-{cue_on}_h1_L2-{h1_lam_l2}_h2_L2-{h2_lam_l2}_sparse_type-{sparse_type}_fb_on-{fb_on}_inh_kappa-{ring_w_kappa_inh}_modnum-{m_idx}'
    
                                    # load the trained model, set to eval, requires_grad == False
                                    net = load_model( f'{fn}.pt',cuda_to_eval )
                                    
                                    # update eval params
                                    net.recurrent_layer.batch_size = n_trials_per_batch
                                    net.recurrent_layer.W_bias_ff_eval_scalar = ffa
                                    net.recurrent_layer.W_bias_fb_eval_scalar = fba
                                    
                                    # store the input weights (stims)
                                    inp_stims[ m_idx,:,: ] = net.recurrent_layer.inp_layer.weight.cpu().detach().numpy()
                                    
                                    # print out some updates and reality check dis type and fb levels
                                    print(f'GPU: {cuda_to_eval}, Task: {task_type}, N_AFC: {n_stim_chans}, Stim: {st}, Delay: {total_delay}, DA: {da}, EA: {ea}, FF: {ffa}, FB: {fba}, Cue On: {cue_on}, Stim Noise: {stim_noise_eval}, h2 L2: {h2_lam_l2}, Model Num: {m_idx}')
                                    
                                    # then load the json to get the scrambled s->r mapping for this model if cue task
                                    if task_type == 'repro_dis_cue':
                                        with open(f'{fn}.json') as f:
                                            pt_info = json.load(f)[1]
                                        
                                        sr_scram = np.full( ( num_cues,n_stim_chans ),np.nan )
                                        for sr in range( num_cues ):
                                            sr_scram[ sr,: ] = pt_info['sr_scram'][f'{sr}']
                                    else:
                                        sr_scram = np.zeros(1)
                                        
                                    # define task    
                                    task = define_task( task_type,T,stim_on,stim_dur,stim_amp,stim_noise_eval,da,ea,n_trials_per_batch,n_stim_chans,num_cues,cue_on,dis_static_dynamic )
                                    
                                    # eval the model at the current dis amp
                                    sp_cor,sp_incor,cl,h1_cor,h1_incor,h2_cor,h2_incor,ff_cor,ff_incor,fb_cor,fb_incor,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,eval_acc[ m_idx ],enough_cor,enough_incor = eval_model_batch( net, task, sr_scram.astype(int), internal_noise_eval, n_stim_chans, num_targ_trials, max_batches )
                                    
                                    # store whether we got enough cor and incor trials for this model
                                    enough_cor_trials[ m_idx ] = enough_cor
                                    enough_incor_trials[ m_idx ] = enough_incor
                                    
                                    # first make sure that max batches wasn't exceeded even for cor trials (if so, don't contine to analyze this model)
                                    if enough_cor == True:
                                    
                                        # store weights
                                        rand_weights[ m_idx,:,: ] = w2
                                        ff_weights[ m_idx,:,: ] = ff_w
                                        fb_weights[ m_idx,:,: ] = fb_w
    
                                        # store indices
                                        params_dict[ m_idx ]['exc1'] = exc1
                                        params_dict[ m_idx ]['exc2'] = exc2
                                        params_dict[ m_idx ]['inh1'] = inh1
                                        params_dict[ m_idx ]['inh2'] = inh2
                                        params_dict[ m_idx ]['tau1'] = tau1
                                        params_dict[ m_idx ]['tau2'] = tau2
            
                                        # get stim presented on each trial
                                        stim_pres_cor = sp_cor[ :,:n_stim_chans ].nonzero()[1]
                                        dis_pres_cor = sp_cor[ :,n_stim_chans: ].nonzero()[1]
                                        if enough_incor == True:
                                            stim_pres_incor = sp_incor[ :,:n_stim_chans ].nonzero()[1]
                                            dis_pres_incor = sp_incor[ :,n_stim_chans: ].nonzero()[1]
                                        
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
                                                cor_avg_ff[ m_idx,s,:,: ] = np.mean( ff_cor[ :,cor_ind,: ],axis=1 )
                                                cor_avg_fb[ m_idx,s,:,: ] = np.mean( fb_cor[ :,cor_ind,: ],axis=1 )
    
                                            if enough_incor == True:
                                                incor_ind = ( stim_pres_incor==s )
                                                if np.sum( incor_ind )>0:
                                                    incor_avg_h1[ m_idx,s,:,: ] = np.mean( h1_incor[ :,incor_ind,: ],axis=1 ) 
                                                    incor_avg_h2[ m_idx,s,:,: ] = np.mean( h2_incor[ :,incor_ind,: ],axis=1 ) 
                                                    incor_avg_ff[ m_idx,s,:,: ] = np.mean( ff_incor[ :,incor_ind,: ],axis=1 )
                                                    incor_avg_fb[ m_idx,s,:,: ] = np.mean( fb_incor[ :,incor_ind,: ],axis=1 )
    
                                        # -------------------------------------
                                        # do the cross-validated decoding for correct trials       
                                        # -------------------------------------
                                        print(f'Decoding correct trials')
                                        cor_acc_exc1[ m_idx,: ],cor_acc_inh1[ m_idx,: ],cor_acc_exc2[ m_idx,: ],cor_acc_inh2[ m_idx,: ] = decode_ls_svm(h1_cor, h2_cor, n_stim_chans, stim_pres_cor, t_step, exc1, inh1, exc2, inh2, train_prcnt, n_cvs, tmpnt_or_cross_tmpnt, num_targ_trials)
                    
                                        if enough_incor == True:
                                            print(f'Decoding incorrect trials')
                                            incor_acc_exc1[ m_idx,: ],incor_acc_inh1[ m_idx,: ],incor_acc_exc2[ m_idx,: ],incor_acc_inh2[ m_idx,: ] = decode_ls_svm(h1_incor, h2_incor, n_stim_chans, stim_pres_incor, t_step, exc1, inh1, exc2, inh2, train_prcnt, n_cvs, tmpnt_or_cross_tmpnt, num_targ_trials)
                                
                                        
                                # save out the data across models
                                np.savez( fn_out,n_tmpts=T,enough_cor_trials=enough_cor_trials,
                                         enough_incor_trials=enough_incor_trials,
                                         cor_acc_exc1=cor_acc_exc1,cor_acc_inh1=cor_acc_inh1,
                                         cor_acc_exc2=cor_acc_exc2,cor_acc_inh2=cor_acc_inh2,
                                         incor_acc_exc1=incor_acc_exc1,incor_acc_inh1=incor_acc_inh1,
                                         incor_acc_exc2=incor_acc_exc2,incor_acc_inh2=incor_acc_inh2, 
                                         eval_acc=eval_acc,rand_weights=rand_weights,ff_weights=ff_weights,
                                         fb_weights=fb_weights,inp_stims=inp_stims,
                                         stim_id_cor=stim_id_cor,all_stims_cor=all_stims_cor,
                                         stim_id_incor=stim_id_incor,all_stims_incor=all_stims_incor,
                                         cor_avg_h1=cor_avg_h1,cor_avg_h2=cor_avg_h2,
                                         cor_avg_ff=cor_avg_ff,cor_avg_fb=cor_avg_fb,
                                         incor_avg_h1=incor_avg_h1,incor_avg_h2=incor_avg_h2,
                                         incor_avg_ff=incor_avg_ff,incor_avg_fb=incor_avg_fb,
                                         params_dict=params_dict )
                    
                            # if file exists
                            else:
                                print(f'Skipping: {fn_out}')
            
    
    
    
    
