#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 2025

@author: johnserences, jserences@ucsd.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch 
import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.stats import sem
import gc

# import class to make model inputs (defines different tasks, and can add other tasks easily using this framework)
from tasks import ctTASKS

def define_task( task_type, T, stim_on, stim_dur, stim_amp, stim_noise, trn_amp, evl_amp, n_trials, n_afc, num_cues, cue_on, dis_static_dynamic ):
    
    #--------------------------------
    # task params
    #--------------------------------
    task_type = task_type             # task type 
    T = T                             # timesteps in each trial
    set_size = 1                      # max stim set size
    n_stim_chans = n_afc              # possible stimulus channels (used to set up net and task)
    stim_amp = stim_amp
    stim_noise = stim_noise
    train_eval = 1                    # training (0, use train_amp for distractors) or evaluating model (1, use eval_amp for distractors)
    train_amp = trn_amp               # since eval, not using this
    eval_amp = evl_amp                # using this amp for eval
    stim_on = stim_on                 # timestep of stimulus onset (set above so both dicts)
    stim_dur = stim_dur               # stim duration
    stim_dis_isi = 0                  # time between stim offset and dis onset
    dis_dur = 20                      # distractor duration
    
    if T == 240:
        num_dis = 6                   # number of distractors - use this to define the delay period on assumption that entire delay occupied with distractors
    else:
        raise ValueError(f'You specified a T value of {T}, but only 180 and 240 are supported')
    
    cue_on = cue_on                   # don't cue until "target" response period (end of delay period)
    cue_dur = T - cue_on
    delay = int( num_dis * dis_dur )  # actual delay period in timesteps
    batch_size = n_trials             # number of trials in each batch
    acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct

    # init dict of task related params 
    settings = {'task' : task_type, 'n_stim_chans' : n_stim_chans, 'set_size' : set_size, 'num_dis' : num_dis, 'dis_static_dynamic' : dis_static_dynamic,
                'T' : T, 'stim_on' : stim_on, 'stim_amp' : stim_amp, 'stim_noise' : stim_noise, 'train_amp' : train_amp, 'eval_amp' : eval_amp,
                'train_eval' : train_eval, 'stim_dur' : stim_dur, 'dis_dur' : dis_dur, 'stim_dis_isi' : stim_dis_isi, 'batch_size' : batch_size,
                'delay' : delay, 'acc_amp_thresh' : acc_amp_thresh, 'num_cues' : num_cues, 'cue_on' : cue_on, 'cue_dur' : cue_dur}

    # create the task object
    return ctTASKS( settings )

#--------------------------------
# load a trained model
#--------------------------------
def load_model( fn, cuda_num ): 
    
    # set the dfault device for all torch stuff... 
    cuda_device = torch.device( f'cuda:{cuda_num}' )
    torch.set_default_device( cuda_device )
    
    # load model to devi
    model = torch.load( fn,map_location=torch.device( f'cuda:{cuda_num}' ),weights_only=False )
    
    # set it to eval (not training)
    model.eval()
    
    # set requires grad = false
    for param in model.parameters():
        param.requires_grad = False
    
    return model

#--------------------------------
# eval trained model (generic version...)
#--------------------------------
def eval_model( model, task, sr_scram, internal_noise ):
    
    # set noise
    model.recurrent_layer.noise = internal_noise
    
    # get the weights
    w1 = model.recurrent_layer.ring_layer.weight.cpu().detach().numpy()
    tmp_w2 = model.recurrent_layer.rand_layer.weight
    m2 = model.recurrent_layer.rand_layer.mask
    w2 = torch.matmul( torch.relu(tmp_w2), m2 ).cpu().detach().numpy()
    ff_w = torch.matmul( model.recurrent_layer.ring_layer.ff_mask,torch.relu( model.recurrent_layer.ring_layer.ff_weight ) ).cpu().detach().numpy()
    fb_w = torch.matmul( model.recurrent_layer.rand_layer.fb_mask,torch.relu( model.recurrent_layer.rand_layer.fb_weight ) ).cpu().detach().numpy()
    rand_mask_diag = torch.diag( model.recurrent_layer.rand_layer.fb_mask ).cpu().detach().numpy()
    
    # get exc and inh units in first layer (ring)
    exc1 = np.where( np.sum( w1,axis=0 )>=0 )[0]
    inh1 = np.where( np.sum( w1,axis=0 )<0 )[0]
    
    # get exc and inh for rand layer...
    exc2 = np.where( rand_mask_diag == 1 )[0]
    inh2 = np.where( rand_mask_diag == 0 )[0]
            
    # generate batch of inputs...
    if task.task == 'repro_dis':
        inp,tmp_sp = task.stim_reproduction_dis() 
        cues = np.zeros( (task.T, task.batch_size, task.n_stim_chans) )   # dummy, not used
        cl = np.zeros( task.batch_size )                                  # cue label, dummy, not used
        
    elif task.task == 'repro_dis_cue':
        inp,cues,sp,cl = task.stim_reproduction_dis_cue()             # also returns real cue inputs and real cue labels
    
    # pass inputs...get outputs from trained model
    with torch.no_grad():
        outputs,h1,h2,ff,fb,tau1,tau2 = model( inp,cues )
        
    # compute eval acc
    if task.task == 'repro_dis':
        tbt_acc,mean_acc = task.compute_acc( outputs,sp ) 
        
    elif task.task == 'repro_dis_cue':
        tbt_acc,mean_acc = task.compute_acc_cue( outputs,sp,sr_scram,cl ) 
        
    print(f'Eval Accuracy: {mean_acc}')
    
    # detach tensors before returning and convert to numpy...
    outputs = outputs.cpu().detach().numpy()
    h1 = h1.cpu().detach().numpy()
    h2 = h2.cpu().detach().numpy()
    ff = ff.cpu().detach().numpy()
    fb = fb.cpu().detach().numpy()
    tau2 = tau2.cpu().detach().numpy()

    return outputs,sp,cl,h1,h2,ff,fb,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,tbt_acc,mean_acc

#----------------------------------------------
# eval a bunch of mini-batches and concatenate to get a target number of 
# correct and incorrect trials
#----------------------------------------------
def eval_model_batch( model, task, sr_scram, internal_noise, n_afc, num_targ_trials, max_batches ):
    
    # set noise
    model.recurrent_layer.noise = internal_noise
    
    # get the weights
    w1 = model.recurrent_layer.ring_layer.weight.cpu().detach().numpy()
    tmp_w2 = model.recurrent_layer.rand_layer.weight
    m2 = model.recurrent_layer.rand_layer.mask
    w2 = torch.matmul( torch.relu(tmp_w2), m2 ).cpu().detach().numpy()
    ff_w = torch.matmul( model.recurrent_layer.ring_layer.ff_mask,torch.relu( model.recurrent_layer.ring_layer.ff_weight ) ).cpu().detach().numpy()
    fb_w = torch.matmul( model.recurrent_layer.rand_layer.fb_mask,torch.relu( model.recurrent_layer.rand_layer.fb_weight ) ).cpu().detach().numpy()
    rand_mask_diag = torch.diag( model.recurrent_layer.rand_layer.fb_mask ).cpu().detach().numpy()
    
    # get exc and inh units in first layer (ring)
    exc1 = np.where( np.sum( w1,axis=0 )>=0 )[0]
    inh1 = np.where( np.sum( w1,axis=0 )<0 )[0]
    
    # get exc and inh for rand layer...
    exc2 = np.where( rand_mask_diag == 1 )[0]
    inh2 = np.where( rand_mask_diag == 0 )[0]
    
    #---------------------------
    # generate batches of inputs until we reach the 
    # target number of correct and incorrect trials
    #---------------------------
    acc = 0
    enough = False
    done_with_cor = False
    done_with_incor = False
    cor_done = np.zeros( n_afc )
    incor_done = np.zeros( n_afc )
    nb = 0
    cor_cnt = 0
    incor_cnt = 0
    
    # start looping to find enough cor and incor trials
    while ( enough == False ) & ( nb < max_batches ):
        
        print(f'Batch num {nb}')
        
        # generate inputs...
        if task.task == 'repro_dis':
            inp,tmp_sp = task.stim_reproduction_dis() 
            cues = np.zeros( (task.T, task.batch_size, task.n_stim_chans) )   # dummy, not used
            cl = np.zeros( task.batch_size )                                  # cue label, dummy, not used
            
        elif task.task == 'repro_dis_cue':
            inp,cues,tmp_sp,cl = task.stim_reproduction_dis_cue()             # also returns real cue inputs and real cue labels
        
        # pass inputs...get outputs from trained model
        with torch.no_grad():
            tmp_outputs,tmp_h1,tmp_h2,tmp_ff,tmp_fb,tau1,tau2 = model( inp,cues )
        
        # compute eval acc
        if task.task == 'repro_dis':
            tmp_tbt_acc, tmp_acc = task.compute_acc( tmp_outputs,tmp_sp ) 
            
        elif task.task == 'repro_dis_cue':
            tmp_tbt_acc,tmp_acc = task.compute_acc_cue( tmp_outputs,tmp_sp,sr_scram,cl ) 
            
        # accumulate acc
        acc += tmp_acc
        print(f'Eval Accuracy on batch {nb}: {tmp_acc}')
    
        # cat model results across batches
        if nb == 0:
            
            # number of possible stims...
            n_stims = tmp_sp.shape[1] // 2
            
            # flatten out the stim on each trial...
            stims = tmp_sp[ :,:n_stims ].nonzero()[1]
            
            # outputs = tmp_outputs
            sp = {}
            h1 = {}
            h2 = {}
            ff = {}
            fb = {}

            # loop over possible stims and fill up dicts...
            for s in range( n_stims ):
                sp[f'cor{s}'] = tmp_sp[ ( tmp_tbt_acc==1 ) & ( stims == s ),: ]
                h1[f'cor{s}'] = tmp_h1[ :, ( tmp_tbt_acc==1 ) & ( stims == s ),: ]
                h2[f'cor{s}'] = tmp_h2[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ]
                ff[f'cor{s}'] = tmp_ff[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ]
                fb[f'cor{s}'] = tmp_fb[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ]
                
                sp[f'incor{s}'] = tmp_sp[ ( tmp_tbt_acc==0 ) & ( stims == s ),: ]
                h1[f'incor{s}'] = tmp_h1[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ]
                h2[f'incor{s}'] = tmp_h2[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ]
                ff[f'incor{s}'] = tmp_ff[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ]
                fb[f'incor{s}'] = tmp_fb[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ]
        
        # start stacking if dicts already made
        else:
            
            # flatten out the stim on each trial...
            stims = tmp_sp[ :,:n_stims ].nonzero()[1]
            
            # loop over stims
            for s in range( n_stims ):
                
                # if we still need more of this stim, then concat, otherwise skip
                if ( h1[f'cor{s}'].shape[1]<num_targ_trials ):
                    sp[f'cor{s}'] = np.vstack( [ sp[f'cor{s}'],tmp_sp[ ( tmp_tbt_acc==1 ) & ( stims == s ),: ] ] )
                    h1[f'cor{s}'] = torch.hstack( [ h1[f'cor{s}'],tmp_h1[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ] ] )
                    h2[f'cor{s}'] = torch.hstack( [ h2[f'cor{s}'],tmp_h2[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ] ] )
                    ff[f'cor{s}'] = torch.hstack( [ ff[f'cor{s}'],tmp_ff[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ] ] )
                    fb[f'cor{s}'] = torch.hstack( [ fb[f'cor{s}'],tmp_fb[ :,( tmp_tbt_acc==1 ) & ( stims == s ),: ] ] )
                else:
                    cor_done[ s ] = 1
                
                # if we still need more of this stim, then concat, otherwise skip
                if ( h1[f'incor{s}'].shape[1]<num_targ_trials ):   
                    sp[f'incor{s}'] = np.vstack( [ sp[f'incor{s}'],tmp_sp[ ( tmp_tbt_acc==0 ) & ( stims == s ),: ] ] )
                    h1[f'incor{s}'] = torch.hstack( [ h1[f'incor{s}'],tmp_h1[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ] ] )
                    h2[f'incor{s}'] = torch.hstack( [ h2[f'incor{s}'],tmp_h2[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ] ] )
                    ff[f'incor{s}'] = torch.hstack( [ ff[f'incor{s}'],tmp_ff[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ] ] )
                    fb[f'incor{s}'] = torch.hstack( [ fb[f'incor{s}'],tmp_fb[ :,( tmp_tbt_acc==0 ) & ( stims == s ),: ] ] )
                else:
                    incor_done[ s ] = 1    
        
        #-----------------------------
        # truncate and stack if we've had enough cor trials - boolean
        # so we don't get in here multiple times if we're still looking for 
        # incor trials
        if np.all( cor_done==1 ) & ( done_with_cor == False ):
            
            # loop over stims to truncate extra if there are any...
            for s in range( n_stims ):
                
                sp[f'cor{s}'] = sp[f'cor{s}'][ :num_targ_trials,: ]
                h1[f'cor{s}']  = h1[f'cor{s}'][ :,:num_targ_trials,: ]
                h2[f'cor{s}'] = h2[f'cor{s}'][ :,:num_targ_trials,: ]
                ff[f'cor{s}'] = ff[f'cor{s}'][ :,:num_targ_trials,: ]
                fb[f'cor{s}'] = fb[f'cor{s}'][ :,:num_targ_trials,: ]
            
            # loop over stims make big arrays that have all stims stacked...
            for s in range( n_stims ):
                
                # init
                if ( s==0 ):
                    sp_cor = sp[f'cor{s}']
                    h1_cor = h1[f'cor{s}']
                    h2_cor = h2[f'cor{s}']
                    ff_cor = ff[f'cor{s}']
                    fb_cor = fb[f'cor{s}']             

                # else concat to stack stims on top of each other. 
                else:
                    sp_cor = np.vstack( [ sp_cor,sp[f'cor{s}'] ] )
                    h1_cor = torch.hstack( [ h1_cor,h1[f'cor{s}'] ] )
                    h2_cor = torch.hstack( [ h2_cor,h2[f'cor{s}'] ] )
                    ff_cor = torch.hstack( [ ff_cor,ff[f'cor{s}'] ] )
                    fb_cor = torch.hstack( [ fb_cor,fb[f'cor{s}'] ] )   
                    
            # set flag so we don't get in here again if still looking for incor trials
            done_with_cor = True

        #-----------------------------
        # truncate and stack if we've had enough incor trials - boolean
        # so we don't get in here multiple times if we're still looking for 
        # cor trials
        if np.all( incor_done==1 ) & ( done_with_incor == False ):
            
            # loop over stims to truncate extra if there are any...
            for s in range( n_stims ):
                
                sp[f'incor{s}'] = sp[f'incor{s}'][ :num_targ_trials,: ]
                h1[f'incor{s}'] = h1[f'incor{s}'][ :,:num_targ_trials,: ]
                h2[f'incor{s}'] = h2[f'incor{s}'][ :,:num_targ_trials,: ]
                ff[f'incor{s}'] = ff[f'incor{s}'][ :,:num_targ_trials,: ]
                fb[f'incor{s}'] = fb[f'incor{s}'][ :,:num_targ_trials,: ]
            
            # loop over stims make big arrays that have all stims stacked...
            for s in range( n_stims ):
                
                # init
                if ( s==0 ):

                    sp_incor = sp[f'incor{s}']
                    h1_incor = h1[f'incor{s}']
                    h2_incor = h2[f'incor{s}']
                    ff_incor = ff[f'incor{s}']
                    fb_incor = fb[f'incor{s}']                

                # else concat to stack stims on top of each other. 
                else:
                    
                    sp_incor = np.vstack( [ sp_incor,sp[f'incor{s}'] ] )
                    h1_incor = torch.hstack( [ h1_incor,h1[f'incor{s}'] ] )
                    h2_incor = torch.hstack( [ h2_incor,h2[f'incor{s}'] ] )
                    ff_incor = torch.hstack( [ ff_incor,ff[f'incor{s}'] ] )
                    fb_incor = torch.hstack( [ fb_incor,fb[f'incor{s}'] ] ) 

            # set flag so we don't get in here again if still looking for incor trials
            done_with_incor = True
            
        #-----------------------------
        # bail on the while loop if we have enough of both
        if ( done_with_cor==True ) & ( done_with_incor==True ):
            enough = True

        #clean up...
        gc.collect()
        torch.cuda.empty_cache()

        #increment num batch counter
        nb += 1

    # compute mean acc over all batches
    acc = acc / nb

    # check to see if max_batches exceeded and set a flag
    # to be returned...also handles the rare case where we hit
    # max_batches and finished filling up both cor and incor
    # on the very last batch
    if nb == max_batches:
        
        # if we did get enough cor trials, just do those...
        if np.all( cor_done==1 ):
            
            # set flags - we know that we didn't get enough incor if we get here..
            enough_cor = True
            
            # detach tensors before returning so in numpy...
            h1_cor = h1_cor.cpu().detach().numpy()
            h2_cor = h2_cor.cpu().detach().numpy()
            ff_cor = ff_cor.cpu().detach().numpy()
            fb_cor = fb_cor.cpu().detach().numpy()
            tau2 = tau2.cpu().detach().numpy()  # these should be the same as incor...            

        # set to false
        else:
            enough_cor = False
             
        # if we did get enough cor trials, just do those...
        if np.all( incor_done==1 ):
            
            # set flags - we know that we didn't get enough incor if we get here..
            enough_incor = True
            
            # detach tensors before returning so in numpy...
            h1_incor = h1_incor.cpu().detach().numpy()
            h2_incor = h2_incor.cpu().detach().numpy()
            ff_incor = ff_incor.cpu().detach().numpy()
            fb_incor = fb_incor.cpu().detach().numpy()
            tau2 = tau2.cpu().detach().numpy()  # these should be the same as cor and might be redundant, but do in both conditionals jic...            
        
        # set to false
        else:
            enough_incor = False
        
        #clean up...
        gc.collect()
        torch.cuda.empty_cache() 
        
        # return both or either just the cor/incor data or nada except the bool flag .
        if ( enough_cor == True ) & ( enough_incor == True ):
            return sp_cor,sp_incor,cl,h1_cor,h1_incor,h2_cor,h2_incor,ff_cor,ff_incor,fb_cor,fb_incor,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,acc,enough_cor,enough_incor
        
        elif ( enough_cor == True ) & ( enough_incor == False ):
            return sp_cor,0,cl,h1_cor,0,h2_cor,0,ff_cor,0,fb_cor,0,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,acc,enough_cor,enough_incor
        
        elif ( enough_cor == False ) & ( enough_incor == True ):
            return 0,sp_incor,cl,0,h1_incor,0,h2_incor,0,ff_incor,0,fb_incor,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,acc,enough_cor,enough_incor
        
        else:
            return 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,enough_cor,enough_incor
    
    # if we found enough cor and incor trials and didn't hit max_batches
    else:
        
        # set flags... 
        enough_cor = True
        enough_incor = True

        # detach tensors before returning so in numpy...
        h1_cor = h1_cor.cpu().detach().numpy()
        h2_cor = h2_cor.cpu().detach().numpy()
        ff_cor = ff_cor.cpu().detach().numpy()
        fb_cor = fb_cor.cpu().detach().numpy()
        tau2 = tau2.cpu().detach().numpy()  # these should be the same as incor...
        
        h1_incor = h1_incor.cpu().detach().numpy()
        h2_incor = h2_incor.cpu().detach().numpy()
        ff_incor = ff_incor.cpu().detach().numpy()
        fb_incor = fb_incor.cpu().detach().numpy()
        
        #clean up...
        gc.collect()
        torch.cuda.empty_cache()    
    
        # return real stuff...
        return sp_cor,sp_incor,cl,h1_cor,h1_incor,h2_cor,h2_incor,ff_cor,ff_incor,fb_cor,fb_incor,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,acc,enough_cor,enough_incor

#----------------------------------------------
# do decoding - either timepoint by timepoint 
# or x-time generalization
#----------------------------------------------
def decode_ls_svm(r_mat, r_mat2, n_afc, stim_val, t_step, exc1, inh1, exc2, inh2, train_prcnt, n_cvs, tmpnt_or_cross_tmpnt, num_targ_trials):
    """
    least square SVM
    
    Parameters
    ----------
    r_mat : matrix of rates (timepoints, neurons)
    
    stim_val : which stimulus was presented on this trial
        
    Returns
    -------
    pred acc for layer 1 and layer 2
        
    """
    
    # num neurons and num timepoints
    n_tmpts = r_mat.shape[0]
    n_trials = r_mat.shape[1]
    trial_split = int( num_targ_trials * train_prcnt )   # for splitting into train and test sets
    
    # separate out the exc and inh units
    h1_exc = r_mat[ :,:,exc1 ]
    h2_exc = r_mat2[ :,:,exc2 ]
    h1_inh = r_mat[ :,:,inh1 ]
    h2_inh = r_mat2[ :,:,inh2 ] 
    
    # num time steps
    n_t_steps = n_tmpts // t_step
    
    # start the cv loop
    for cv in range( n_cvs ):
            
        trn_ind = []
        tst_ind = []
        for s in range( n_afc ):
            s_ind = np.where( stim_val==s )[0]
            rnd_ind = np.random.permutation( len( s_ind ) )
            trn_ind.append( s_ind[ rnd_ind[ :trial_split ] ].tolist() )
            tst_ind.append( s_ind[ rnd_ind[ trial_split: ] ].tolist() )
        
        # flatten
        trn_ind = [ ind for inds in trn_ind for ind in inds ]
        tst_ind = [ ind for inds in tst_ind for ind in inds ]

        # generate training sets
        h1_train_exc = h1_exc[ :,trn_ind,: ]
        h1_train_inh = h1_inh[ :,trn_ind,: ]
        h2_train_exc = h2_exc[ :,trn_ind,: ]
        h2_train_inh = h2_inh[ :,trn_ind,: ]
        
        # generate test sets
        h1_test_exc = h1_exc[ :,tst_ind,: ]
        h1_test_inh = h1_inh[ :,tst_ind,: ]
        h2_test_exc = h2_exc[ :,tst_ind,: ]
        h2_test_inh = h2_inh[ :,tst_ind,: ]
        
        # train and test labels
        train_lab = stim_val[ trn_ind ]
        test_lab = stim_val[ tst_ind ]
            
        # if training/testing separately on each timepoint
        if tmpnt_or_cross_tmpnt == 0: 
        
            # alloc to store the circ corr between actual and pred
            if cv == 0:
                pred_acc_exc1 = np.full( ( n_cvs,n_t_steps ),np.nan )
                pred_acc_inh1 = np.full( ( n_cvs,n_t_steps ),np.nan )
                pred_acc_exc2 = np.full( ( n_cvs,n_t_steps ),np.nan )
                pred_acc_inh2 = np.full( ( n_cvs,n_t_steps ),np.nan )
        
        elif tmpnt_or_cross_tmpnt == 1: 
        
            # alloc to store the circ corr between actual and pred
            if cv == 0:
                pred_acc_exc1 = np.full( ( n_cvs,n_t_steps,n_t_steps ),np.nan )
                pred_acc_inh1 = np.full( ( n_cvs,n_t_steps,n_t_steps ),np.nan )
                pred_acc_exc2 = np.full( ( n_cvs,n_t_steps,n_t_steps ),np.nan )
                pred_acc_inh2 = np.full( ( n_cvs,n_t_steps,n_t_steps ),np.nan )        
        
        # only go if there is at least one sample of each stim in each split
        if ( len( np.unique( train_lab ) ) == n_afc ) & ( len( np.unique( test_lab ) ) == n_afc ):
        
            # if training/testing separately on each timepoint
            if tmpnt_or_cross_tmpnt == 0: 
                
                # loop over timepoints 
                for t_idx,t in enumerate( range( 0,n_tmpts,t_step ) ) :
                 
                    # (re)initialize models just to be safe
                    ls_svm_model_h1_exc = RidgeClassifier( class_weight = 'balanced' )
                    ls_svm_model_h1_inh = RidgeClassifier( class_weight = 'balanced' )
                    ls_svm_model_h2_exc = RidgeClassifier( class_weight = 'balanced' )    
                    ls_svm_model_h2_inh = RidgeClassifier( class_weight = 'balanced' )
        
                    # hidden layer 1
                    
                    # fit the ls svm ofr h1 exc
                    ls_svm_model_h1_exc.fit( h1_train_exc[ t, : ],train_lab )
                    
                    # compute acc for h1 exc
                    pred_acc_exc1[ cv,t_idx ]  = ls_svm_model_h1_exc.score( h1_test_exc[ t, : ],test_lab )
    
                    # fit the h1 inh model
                    ls_svm_model_h1_inh.fit( h1_train_inh[ t, : ],train_lab )
                    
                    # compute acc for h1
                    pred_acc_inh1[ cv,t_idx ]  = ls_svm_model_h1_inh.score( h1_test_inh[ t, : ],test_lab )
        
                    # hidden layer 2
                    
                    # fit the ls svm ofr h1 exc
                    ls_svm_model_h2_exc.fit( h2_train_exc[ t, : ],train_lab )
                    
                    # compute acc for h1 exc
                    pred_acc_exc2[ cv,t_idx ]  = ls_svm_model_h2_exc.score( h2_test_exc[ t, : ],test_lab )
    
                    # fit the h2 inh model
                    ls_svm_model_h2_inh.fit( h2_train_inh[ t, : ],train_lab )
                    
                    # compute acc for h1
                    pred_acc_inh2[ cv,t_idx ]  = ls_svm_model_h2_inh.score( h2_test_inh[ t, : ],test_lab )
        
            # generalize across time...
            elif tmpnt_or_cross_tmpnt == 1: 
                    
                # outer loop over timepoints 
                for t_idx,t in enumerate( range( 0,n_tmpts,t_step ) ) :
        
                    # (re)initialize models just to be safe
                    ls_svm_model_h1_exc = RidgeClassifier( class_weight = 'balanced' )
                    ls_svm_model_h1_inh = RidgeClassifier( class_weight = 'balanced' )
                    ls_svm_model_h2_exc = RidgeClassifier( class_weight = 'balanced' )    
                    ls_svm_model_h2_inh = RidgeClassifier( class_weight = 'balanced' )
                    
                    # fit the models
                    ls_svm_model_h1_exc.fit( h1_train_exc[ t,:,: ], train_lab )
                    ls_svm_model_h1_inh.fit( h1_train_inh[ t,:,: ], train_lab )
                    ls_svm_model_h2_exc.fit( h2_train_exc[ t,:,: ], train_lab )
                    ls_svm_model_h2_inh.fit( h2_train_inh[ t,:,: ], train_lab )
        
                    # inner loop over timepoints 
                    for tt_idx,tt in enumerate( range( 0,n_tmpts,t_step ) ) :           
                        
                        # compute acc for h1
                        pred_acc_exc1[ cv,t_idx,tt_idx ] = ls_svm_model_h1_exc.score( h1_test_exc[ tt,:,: ],test_lab )
                        pred_acc_inh1[ cv,t_idx,tt_idx ] = ls_svm_model_h1_inh.score( h1_test_inh[ tt,:,: ],test_lab )
                        
                        # compute acc for h2
                        pred_acc_exc2[ cv,t_idx,tt_idx ] = ls_svm_model_h2_exc.score( h2_test_exc[ tt,:,: ],test_lab )
                        pred_acc_inh2[ cv,t_idx,tt_idx ] = ls_svm_model_h2_inh.score( h2_test_inh[ tt,:,: ],test_lab )
        
        # if training and testing sets don't have n_afc unique labels then 
        # throw an error because that shouldn't happen...
        else:
            raise ValueError('Train/Test do not have n_afc labels')
              
    # return mean pred acc across all cv folds
    return np.nanmean( pred_acc_exc1,axis=0 ), np.nanmean( pred_acc_inh1,axis=0 ), np.nanmean( pred_acc_exc2,axis=0 ), np.nanmean( pred_acc_inh2,axis=0 )


def pooled_covariance_matrix( data ):
    """
    Pooled covariance matrix across groups

    Parameters
    ----------
        data (list of ndarrays): A list where each element is a
        numpy array with samples as rows
        and features as columns.

    Returns
    -------
        Pooled covariance matrix
    """
    
    nf = data[0].shape[1]
    pooled_cov = np.zeros( ( nf, nf ) )
    total_df = 0

    # loop over data from each condition
    for cond_data in data:
        n_trials = cond_data.shape[0]
        if n_trials > 1:  # Need at least 2 samples 
            group_cov = np.cov(cond_data, rowvar=False)  # rowvar=False for columns as variables
            df = n_trials - 1
            pooled_cov += df * group_cov
            total_df += df

    if total_df > 0:
        return pooled_cov / total_df
    
    else:
        raise ValueError("Not enough data to calculate pooled covariance matrix.")


def compute_dist(r_mat, r_mat2, n_afc, stim_val, t_step, dist_metric):
    """
    
    Compute distance between activity patterns...
    
    Parameters
    ----------
    r_mat : matrix of rates (timepoints, neurons)
    
    stim_val : which stimulus was presented on this trial
        
    Returns
    -------
    distance....
        
    """
    
    # num neurons and num timepoints
    n_tmpts = r_mat.shape[0]
    n_trials = r_mat.shape[1]
    trial_split = n_trials // 2   # for splitting into train and test sets
    
    # num time steps
    n_t_steps = n_tmpts // t_step
    
    
    if dist_metric == 'correlation': 
    
        # alloc to store the circ corr between actual and pred
        pred_acc = np.zeros( ( n_t_steps,n_t_steps ) )    
        pred_acc2 = np.zeros( ( n_t_steps,n_t_steps ) ) 
        
        # outer loop over timepoints 
        for t_idx,t in enumerate( range( 0,n_tmpts,t_step ) ) :

            # train test split
            rnd_tri = np.random.permutation( n_trials )
            
            h1_split1 = r_mat[t,rnd_tri[:trial_split],:]
            h2_split1 = r_mat2[t,rnd_tri[:trial_split],:]
            y_split1 = stim_val[rnd_tri[:trial_split]]
            
            # inner loop over timepoints 
            for tt_idx,tt in enumerate( range( 0,n_tmpts,t_step ) ) :           
                dum = 42
                
    return pred_acc, pred_acc2


def do_pca( all_data, exc, inh, n_components, task, trn_amp, evl_amp ):

    # Set the number of components
    pca=PCA( n_components=n_components )
    
    # transpose the [timepoints, trials, units] synpatic current data from all units
    ad = np.transpose( all_data.copy(), (1, 0, 2) ) # trials x timepoints x units
    
    # store trials, timepoints, neurons
    n_trials = ad.shape[0]
    n_tmpts = ad.shape[1]
    n_neus = ad.shape[2]
    
    # expand across trials and timepoints
    X = ad.reshape( (n_trials*n_tmpts, n_neus) )
    
    # fit/transform data with all units
    pca_all = pca.fit_transform( X )
    pca_all = pca_all.reshape( (n_trials,n_tmpts,3) )
    
    # transpose the synpatic current data from exc units
    ed = np.transpose( exc.copy(), (1, 0, 2) ) # trials x timepoints x units
    
    # store trials, timepoints, neurons
    n_trials = ed.shape[0]
    n_tmpts = ed.shape[1]
    n_neus = ed.shape[2]
    
    # expand across trials and timepoints
    X = ed.reshape( (n_trials*n_tmpts, n_neus) )
    
    # fit/transform h1 data with all units
    pca_exc = pca.fit_transform( X )
    pca_exc = pca_exc.reshape( (n_trials,n_tmpts,3) )
    
    # transpose the synpatic current data from inh units
    ind = np.transpose( inh.copy(), (1, 0, 2) ) # trials x timepoints x units
    
    # store trials, timepoints, neurons
    n_trials = ind.shape[0]
    n_tmpts = ind.shape[1]
    n_neus = ind.shape[2]
    
    # expand across trials and timepoints
    X = ind.reshape( (n_trials*n_tmpts, n_neus) )
    
    # fit/transform data with inh units
    pca_inh = pca.fit_transform( X )
    pca_inh = pca_inh.reshape( (n_trials,n_tmpts,3) )
    
    return pca_all, pca_exc, pca_inh

# helper to plot PCA trajs
def plot_pca( pca_all, pca_exc, pca_inh, tri_type, plt_tbt_avg, ax_range, t_lim, cols, alpha, markersize, elev, azim ):
    
    # set up the figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5),
                        subplot_kw={'projection': '3d'})
    
    # set up viewpoint...
    for a in range( 3 ):
        axs[a].view_init(elev=elev, azim=azim, roll=0)

    # plot trial by trial
    if plt_tbt_avg == 0:
        
        n_trials = pca_all.shape[0]
        
        # loop over trials 
        for t in range( n_trials ):
            axs[0].plot( pca_all[t,:t_lim,0], pca_all[t,:t_lim,1], pca_all[t,:t_lim,2],cols[tri_type[t]],alpha=alpha )
            axs[1].plot( pca_exc[t,:t_lim,0], pca_exc[t,:t_lim,1], pca_exc[t,:t_lim,2],cols[tri_type[t]],alpha=alpha )
            axs[2].plot( pca_inh[t,:t_lim,0], pca_inh[t,:t_lim,1], pca_inh[t,:t_lim,2],cols[tri_type[t]],alpha=alpha )

            # markers
            axs[0].plot( pca_all[t,0,0],pca_all[t,0,1],pca_all[t,0,2],f'{cols[tri_type[t]]}o', markersize=markersize )
            axs[0].plot( pca_all[t,t_lim-1,0],pca_all[t,t_lim-1,1],pca_all[t,t_lim-1,2],f'{cols[tri_type[t]]}v', markersize=markersize )
            axs[1].plot( pca_exc[t,0,0],pca_exc[t,0,1],pca_exc[t,0,2],f'{cols[tri_type[t]]}o', markersize=markersize )
            axs[1].plot( pca_exc[t,t_lim-1,0],pca_exc[t,t_lim-1,1],pca_exc[t,t_lim-1,2],f'{cols[tri_type[t]]}v', markersize=markersize )
            axs[2].plot( pca_inh[t,0,0],pca_inh[t,0,1],pca_inh[t,0,2],f'{cols[tri_type[t]]}o', markersize=markersize )
            axs[2].plot( pca_inh[t,t_lim-1,0],pca_inh[t,t_lim-1,1],pca_inh[t,t_lim-1,2],f'{cols[tri_type[t]]}v', markersize=markersize )
        
        for a in range( 3 ):
            axs[a].set_xlim( xmin=-ax_range[0], xmax=ax_range[0] )
            axs[a].set_ylim( ymin=-ax_range[1], ymax=ax_range[1] )
            axs[a].set_zlim( zmin=-ax_range[2], zmax=ax_range[2] )        
    
    # plot avg by condition
    elif plt_tbt_avg == 1:

        # how many conditions? 
        n_conds = len( np.unique( tri_type ) )
        
        # loop over conditions to avg
        for c in range( n_conds ):

            # compute and plot avg trajs
            all = np.mean( pca_all[tri_type==c,:,:],axis=0 )
            exc = np.mean( pca_exc[tri_type==c,:,:],axis=0 )
            inh = np.mean( pca_inh[tri_type==c,:,:],axis=0 )
            axs[0].plot( all[:t_lim,0],all[:t_lim,1], all[:t_lim,2],cols[c], alpha=alpha )
            axs[1].plot( exc[:t_lim,0],exc[:t_lim,1], exc[:t_lim,2],cols[c], alpha=alpha )
            axs[2].plot( inh[:t_lim,0],inh[:t_lim,1], inh[:t_lim,2],cols[c], alpha=alpha )

            # plot start/stop markers
            axs[0].plot( all[0,0],all[0,1],all[0,2],f'{cols[c]}o', markersize=markersize )
            axs[0].plot( all[t_lim-1,0],all[t_lim-1,1],all[t_lim-1,2],f'{cols[c]}v', markersize=markersize )
            axs[1].plot( exc[0,0],exc[0,1],exc[0,2],f'{cols[c]}o', markersize=markersize )
            axs[1].plot( exc[t_lim-1,0],exc[t_lim-1,1],exc[t_lim-1,2],f'{cols[c]}v', markersize=markersize )
            axs[2].plot( inh[0,0],inh[0,1],inh[0,2],f'{cols[c]}o', markersize=markersize )
            axs[2].plot( inh[t_lim-1,0],inh[t_lim-1,1],inh[t_lim-1,2],f'{cols[c]}v', markersize=markersize )

        for a in range( 3 ):
            axs[a].set_xlim( xmin=-ax_range[0], xmax=ax_range[0] )
            axs[a].set_ylim( ymin=-ax_range[1], ymax=ax_range[1] )
            axs[a].set_zlim( zmin=-ax_range[2], zmax=ax_range[2] )

    # show
    plt.title('Start = O, End = V')
    plt.show()
    
def traj_dist( X ):
    
    '''
    compute mean euc distance...
    '''
    
    return np.sum( np.sqrt( np.sum( np.diff(X,axis=0)**2,axis=1 ) ) ) / X.shape[0]
    
def maha_class( X_trn, y_trn, X_tst, y_tst ):
    '''
        js 08222024
        
        Computes two-class classification accuracy based on the Mahalanobis distance 
        between each pattern in the test data and the mean patterns associated
        with each condition in independent training data
         
        input:
           X_trn = [num observation, num variable] matrix of data
           y_trn = [num observation,] vector of condition labels for each trial
           X_tst = samesies as X_trn but test data
           y_tst = samesies as y_trn but test data
    '''
    
    u_conds = np.unique( y_trn )            # unique conditions
    n_conds = len( u_conds )                # number of conditions
    
    # mean of each class and store counts for each class
    # and compute the pooled covariance matrix
    m_trn = np.full( ( n_conds,X_trn.shape[1] ), np.nan )
    c_cnts = np.full( n_conds,np.nan )
    S = np.zeros( ( X_trn.shape[1], X_trn.shape[1] ) )

    for c_idx, cond in enumerate( u_conds ):
        Xc = X_trn[ y_trn==cond,: ]
        m_trn[ c_idx,: ] = np.mean( Xc,axis=0 )
        c_cnts[ c_idx ] = np.count_nonzero( y_trn==cond )
        S += ( c_cnts[ c_idx ]-1 ) * np.cov( Xc,rowvar=False )#( np.cov( Xc,rowvar=False ) / ( c_cnts[ c_idx ]-1 ) )
    
    # norm cov by sum of n-1 for each cond
    S *= ( 1/np.sum(c_cnts-1) )  
    
    # inv of cov matrix - using pinv
    # and not inv cause some cov matrices
    # are low-rank...not ideal, but...
    invS = np.linalg.pinv(S)

    # compute class as argmin weighted distance from each
    # condition mean (mahalanobis distance)
    dist = np.full( (n_conds,X_tst.shape[0]),np.nan )
    for cond in range( n_conds ):
        dist[ cond,: ] = np.diag( ( X_tst - m_trn[ cond,: ] ) @ invS @ ( X_tst - m_trn[ cond,: ] ).T )

    # get predicted class label for each trial
    actual = y_tst
    pred = np.argmin( dist,axis=0 )
    acc = np.sum( actual == pred ) / len( y_tst )

    # return acc and dicts
    return acc, actual, pred

# plt mean response from inh units tuned to stim and from units
# tuned ortho to the stim
def plot_act( d2p, T, avg_win, cval, u_type, legend_lab, title_lab, axs  ):
    
    # x-axis for plotting
    x = np.arange( 0,T )
    
    # plt mean response from inh units tuned to stim and from units
    # tuned ortho to the stim

    middle = d2p.shape[2] // 2
    md_stim = np.mean( np.mean( d2p[ :,:,middle-avg_win:middle+avg_win+1 ],axis=0 ),axis=1 )   # mean data in units tuned to stim
    semd_stim = sem( np.mean( d2p[ :,:,middle-avg_win:middle+avg_win+1 ],axis=2 ),axis=0 ) 
    md_ortho = np.mean( ( np.mean( d2p[ :,:,:avg_win+1 ],axis=2 ) + np.mean( d2p[ :,:,-avg_win: ],axis=2 ) ) / 2,axis=0 )   # mean data in units tuned to stim
    semd_ortho = sem( ( np.mean( d2p[ :,:,:avg_win+1 ],axis=2 ) + np.mean( d2p[ :,:,-avg_win: ],axis=2 ) ) / 2,axis=0 ) 
    axs.plot(x, md_stim, c=cval, lw=2, label=legend_lab)
    axs.fill_between(x, md_stim - semd_stim, md_stim + semd_stim, color=cval, alpha=0.3)
    axs.plot(x, md_ortho, lw=2, ls='--', c=cval)
    axs.fill_between(x, md_ortho - semd_ortho, md_ortho + semd_ortho, color=cval, alpha=0.3)

    # Add labels and legend
    axs.set_ylim([-0.1,1.1])
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Response')
    axs.set_title(f'{u_type} Units in h1, DisAmp: {title_lab}')


# helper to plot full set of xgen matrices
def plot_xgen( avg_acc, u_type, n_afc, dis_amp, fig, axs  ):

    im = axs.imshow( avg_acc,interpolation=None,origin='lower',vmin=0,vmax=1 )
    fig.colorbar(im,ax=axs)
    axs.set_title(f'N_AFC: {n_afc}, {u_type}, DisAmp: {dis_amp}')
    
# helper to plot evl_amp = trn_amp
def plot_xgen_equal( avg_acc,color_map ):

    # number of training dis amp levels
    n_trn_amps = avg_acc.shape[0]
    
    # set up the figure
    fig, axs = plt.subplots(ncols=n_trn_amps, nrows=1, figsize=(10, 5))

    for trn in range( n_trn_amps ):
        ax = axs[trn]
        im = ax.imshow( avg_acc[ trn,:,: ],interpolation=None,origin='lower',vmin=0,vmax=1,cmap=color_map )
        ax.set_xlabel('Test Timepoint')
        ax.set_ylabel('Train Timepoint')
        
    # fig.colorbar(im,ax=ax)   
    plt.tight_layout()
    plt.show()

    
        