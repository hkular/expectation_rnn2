#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 12:30:07 2025

@author: johnserences
"""

import numpy as np
import matplotlib.pyplot as plt
import torch 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import RidgeClassifier
import gc

# load the model, turn off grads
def load_model( fn,device ): 
    
    # load model...
    model = torch.load(fn,map_location=torch.device(device), weights_only=False)

    # set it to eval (not training)
    model.eval()
    
    # set requires grad = false
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def load_model_cuda( fn, cuda_num ):
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


def load_model_mps(fn):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend is not available on this machine.")
    
    mps_device = torch.device("mps")
    torch.set_default_device(mps_device)
    
    # Step 1: load to CPU so CUDA storage keys get translated
    model = torch.load(fn, map_location=torch.device("cpu"), weights_only=False)
    
    # Step 2: move model to MPS
    model.to(mps_device)
    
    # Step 3: set eval mode and disable grads
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    
    return model
# eval a batch of trials with a trained model
def eval_model( model, task, sr_scram ):
    
    if task.task == 'rdk_reproduction':
        # get a batch of inputs and targets
        inputs,s_label = task.generate_rdk_reproduction_stim()  
        
        #inputs = inputs.cpu()
        targets = task.generate_rdk_reproduction_target( s_label )
    
        # pass inputs...get outputs and hidden layer states if 
        # desired
        with torch.no_grad():
            outputs,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3 = model( inputs )
        
        # compute eval acc
        m_acc,tbt_acc = task.compute_acc_reproduction( outputs,s_label ) 

    elif task.task == 'rdk':
        
        # get a batch of inputs and targets
        inputs,s_label = task.generate_rdk_stim()  
        
        #inputs = inputs.cpu()
        targets = task.generate_rdk_target( s_label )
    
        # pass inputs...get outputs and hidden layer states if 
        # desired
        with torch.no_grad():
            outputs,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3 = model( inputs )
        
        # compute eval acc
        m_acc,tbt_acc = task.compute_acc(outputs,s_label)
        
    elif task.task == 'rdk_repro_cue':
        
        # get a batch of inputs and targets
        inputs,cues,s_label,c_label = task.generate_rdk_reproduction_cue_stim()
        
        # plot inputs
        #plt.plot(inputs[:,0,:])
        #plt.axvline(100, color='black', linestyle='--', linewidth=2, label='Cue onset')
    
             
        targets = task.generate_rdk_reproduction_cue_target( s_label, sr_scram, c_label )
    
        # pass inputs...get outputs and hidden layer states if 
        # desired
        with torch.no_grad():
            outputs,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3 = model( inputs,cues )
        
        # compute eval acc
        m_acc,tbt_acc = task.compute_acc_reproduction_cue(outputs,s_label,targets) 
    
    print(f'Eval Accuracy: {m_acc}')
    
    # detach tensors
    inputs = inputs.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    h1 = h1.cpu().detach().numpy()
    h2 = h2.cpu().detach().numpy()
    h3 = h3.cpu().detach().numpy()
    ff12 = ff12.cpu().detach().numpy()
    ff23 = ff23.cpu().detach().numpy()
    fb21 = fb21.cpu().detach().numpy()
    fb32 = fb32.cpu().detach().numpy()
    tau1 = tau1.cpu().detach().numpy()
    tau2 = tau2.cpu().detach().numpy()
    tau3 = tau3.cpu().detach().numpy()
    
    # get h layer weights
    w1 = model.recurrent_layer.h_layer1.weight.cpu().detach().numpy()
    w2 = model.recurrent_layer.h_layer2.weight.cpu().detach().numpy()
    w3 = model.recurrent_layer.h_layer3.weight.cpu().detach().numpy()
    
    # get exc and inh units
    exc1 = np.where( np.sum( w1, axis =0 )>= 0 )[0]    
    inh1 = np.where( np.sum( w1, axis =0 )< 0 )[0]  
    exc2 = np.where( np.sum( w2, axis =0 )>= 0 )[0]    
    inh2 = np.where( np.sum( w2, axis =0 )< 0 )[0]  
    exc3 = np.where( np.sum( w3, axis =0 )>= 0 )[0]    
    inh3 = np.where( np.sum( w3, axis =0 )< 0 )[0]  

    return outputs,s_label,w1,w2,w3,exc1,inh1,exc2,inh2,exc3,inh3,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau1,tau3,m_acc,tbt_acc,cues

from cue_layer import *
from inp_layer import *
#----------------------------------------------
# measure feedback contribution to determine 
# what to scale down to
#----------------------------------------------
def measure_feedback_contributions(model, stims, cues, fb_scalars, rnn_settings):
    """
    Measure relative contribution of feedback currents in each layer of your 3-layer RNN.

    Parameters
    ----------
    model : your RNN model (with h_layer1, h_layer2, h_layer3 defined)
    inputs : torch.Tensor (batch x time x input_dim)
    fb_scalars : list of floats, scaling factors to apply to feedback weights

    Returns
    -------
    results : dict
        Dictionary of {fb_scalar: {layer1: ratio, layer2: ratio}}
        where ratio = ||I_fb|| / ||I_total|| averaged over time & batch.
    """

    results = {}
    batch_size = stims.shape[1]
    n_time = stims.shape[0]
    
    device = stims.device
    
    cue_layer = CueLayer( rnn_settings )
    inp_layer = InpLayer( rnn_settings )
    apply_dale = rnn_settings['apply_dale']
    noise = rnn_settings['noise']
    dt = rnn_settings['dt']
    
    for alpha in fb_scalars:
        fb_frac_layer1 = 0.0
        fb_frac_layer2 = 0.0
    
        # Initialize hidden states (small random numbers, as in forward)
        next_x1 = torch.randn(batch_size, model.recurrent_layer.h_size1, device=device)/100
        next_x2 = torch.randn(batch_size, model.recurrent_layer.h_size2, device=device)/100
        next_x3 = torch.randn(batch_size, model.recurrent_layer.h_size3, device=device)/100
    
        r1 = model.recurrent_layer.act_func(next_x1)
        r2 = model.recurrent_layer.act_func(next_x2)
        r3 = model.recurrent_layer.act_func(next_x3)
    
        # scale taus
        h_taus_sig1 = torch.sigmoid(model.recurrent_layer.h_layer1.h_taus_gaus)*(model.recurrent_layer.h_tau1[1]-model.recurrent_layer.h_tau1[0])+model.recurrent_layer.h_tau1[0]
        h_taus_sig2 = torch.sigmoid(model.recurrent_layer.h_layer2.h_taus_gaus)*(model.recurrent_layer.h_tau2[1]-model.recurrent_layer.h_tau2[0])+model.recurrent_layer.h_tau2[0]
        h_taus_sig3 = torch.sigmoid(model.recurrent_layer.h_layer3.h_taus_gaus)*(model.recurrent_layer.h_tau3[1]-model.recurrent_layer.h_tau3[0])+model.recurrent_layer.h_tau3[0]
    
        for t in range(n_time):
            x_t = stims[t]
    
            # cue input if needed
            cue1 = 0
            cue2 = 0
            cue3 = cue_layer(cues[t]) 
    
            # --- Layer 1 ---
            stim_inp = inp_layer(x_t)
            I_rec1 = model.recurrent_layer.h_layer1(r1)
            fb_21 = model.recurrent_layer.h_layer2.mfb_21 @ model.recurrent_layer.h_layer2.wfb_21.T
            fb_21 = torch.matmul(r2, torch.relu(fb_21) if apply_dale[1] else fb_21)
            fb_21_scaled = alpha * fb_21
    
            I_total1 = stim_inp + I_rec1 + fb_21_scaled + cue1
    
            fb_frac_layer1 += (torch.norm(fb_21_scaled, dim=1) / (torch.norm(I_total1, dim=1)+1e-8)).mean().item()
    
            next_x1 = (1-(dt/h_taus_sig1))*next_x1 + (dt/h_taus_sig1)*I_total1
            next_x1 += torch.randn_like(next_x1) * noise
            r1 = torch.sigmoid(next_x1)
    
            # --- Layer 2 ---
            ff_12 = model.recurrent_layer.h_layer1.mff_12 @ model.recurrent_layer.h_layer1.wff_12.T
            ff_12 = torch.matmul(r1, torch.relu(ff_12) if apply_dale[0] else ff_12)
    
            fb_32 = model.recurrent_layer.h_layer3.mfb_32 @ model.recurrent_layer.h_layer3.wfb_32.T
            fb_32 = torch.matmul(r3, torch.relu(fb_32) if apply_dale[2] else fb_32)
            fb_32_scaled = alpha * fb_32
    
            I_rec2 = model.recurrent_layer.h_layer2(r2)
            I_total2 = ff_12 + I_rec2 + fb_32_scaled + cue2
    
            fb_frac_layer2 += (torch.norm(fb_32_scaled, dim=1) / (torch.norm(I_total2, dim=1)+1e-8)).mean().item()
    
            next_x2 = (1-(dt/h_taus_sig2))*next_x2 + (dt/h_taus_sig2)*I_total2
            next_x2 += torch.randn_like(next_x2) * noise
            r2 = torch.sigmoid(next_x2)
    
            # --- Layer 3 ---
            ff_23 = model.recurrent_layer.h_layer2.mff_23 @ model.recurrent_layer.h_layer2.wff_23.T
            ff_23 = torch.matmul(r2, torch.relu(ff_23) if apply_dale[1] else ff_23)
    
            I_rec3 = model.recurrent_layer.h_layer3(r3)
            I_total3 = ff_23 + I_rec3 + cue3
            next_x3 = (1-(dt/h_taus_sig3))*next_x3 + (dt/h_taus_sig3)*I_total3
            next_x3 += torch.randn_like(next_x3) * noise
            r3 = torch.sigmoid(next_x3)
    
        # Average over time
        results[alpha] = {
            'layer1': fb_frac_layer1 / n_time,
            'layer2': fb_frac_layer2 / n_time
        }
    
    return results


#----------------------------------------------
# eval a bunch of mini-batches and concatenate to get a target number of
# correct and incorrect trials
#----------------------------------------------
def eval_model_batch( model, task, sr_scram, internal_noise, n_afc, num_targ_trials ):
    # set noise
    model.recurrent_layer.noise = internal_noise
    # get the weights
    w1 = model.recurrent_layer.h_layer1.weight.cpu().detach().numpy()
    w2 = torch.matmul(model.h_layer1.mff_12, torch.relu(model.h_layer1.wff_12)).cpu().detach().numpy()
    w3 = torch.matmul(model.h_layer2.mff_23, torch.relu(model.h_layer2.wff_23)).cpu().detach().numpy()
    torch.matmul(model.h_layer2.mfb_21, torch.relu(model.h_layer2.wfb_21))
    ff12_w = torch.matmul( model.recurrent_layer.h_layer1.mff12,torch.relu( model.recurrent_layer.h_layer1.wff12) ).cpu().detach().numpy()
    fb21_w = torch.matmul( model.recurrent_layer.h_layer2.mfb21,torch.relu( model.recurrent_layer.h_layer2.wfb21 ) ).cpu().detach().numpy()
    ff23_w = torch.matmul( model.recurrent_layer.h_layer2.mff23,torch.relu( model.recurrent_layer.h_layer2.wff23 ) ).cpu().detach().numpy()
    fb32_w = torch.matmul( model.recurrent_layer.h_layer3.mfb32,torch.relu( model.recurrent_layer.h_layer3.wfb32 ) ).cpu().detach().numpy()
    h2_mask_diag = torch.diag( model.recurrent_layer.h_layer2.mfb21 ).cpu().detach().numpy()
    h3_mask_diag = torch.diag( model.recurrent_layer.h_layer3.mfb32 ).cpu().detach().numpy()
    # get exc and inh units in h layers
    exc1 = np.where( np.sum( w1,axis=0 )>=0 )[0]
    inh1 = np.where( np.sum( w1,axis=0 )<0 )[0]
    exc2 = np.where( h2_mask_diag == 1 )[0]
    inh2 = np.where( h2_mask_diag == 0 )[0]
    exc2 = np.where( h3_mask_diag == 1 )[0]
    inh2 = np.where( h3_mask_diag == 0 )[0]
    #---------------------------
    # generate batches of inputs until we reach the
    # target number of correct and incorrect trials
    #---------------------------
    acc = 0
    enough = False
    cor_done = np.zeros( n_afc )
    incor_done = np.zeros( n_afc )
    nb = 0
    cor_cnt = 0
    incor_cnt = 0
    while ( enough == False ):
        print(f'Batch num {nb}')
        # generate inputs...
        if task.task == 'rdk_repro':
            inp,tmp_sp = task.generate_rdk_stim()
            cues = np.zeros( (task.T, task.batch_size, task.n_stim_chans) )   # dummy, not used
            cl = np.zeros( task.batch_size )                                  # cue label, dummy, not used
        elif task.task == 'rdk_repro_cue':
            inp,cues,tmp_sp,cl = task.generate_rdk_reproduction_cue_stim()             # also returns real cue inputs and real cue labels
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
        # start stacking
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
        # bail and truncate if we've had enough...
        if ( np.all( cor_done==1 ) ) & ( np.all( incor_done==1 ) ):
            # loop over stims to truncate extra if there are any...
            for s in range( n_stims ):
                sp[f'cor{s}'] = sp[f'cor{s}'][ :num_targ_trials,: ]
                h1[f'cor{s}']  = h1[f'cor{s}'][ :,:num_targ_trials,: ]
                h2[f'cor{s}'] = h2[f'cor{s}'][ :,:num_targ_trials,: ]
                ff[f'cor{s}'] = ff[f'cor{s}'][ :,:num_targ_trials,: ]
                fb[f'cor{s}'] = fb[f'cor{s}'][ :,:num_targ_trials,: ]
                sp[f'incor{s}'] = sp[f'incor{s}'][ :num_targ_trials,: ]
                h1[f'incor{s}'] = h1[f'incor{s}'][ :,:num_targ_trials,: ]
                h2[f'incor{s}'] = h2[f'incor{s}'][ :,:num_targ_trials,: ]
                ff[f'incor{s}'] = ff[f'incor{s}'][ :,:num_targ_trials,: ]
                fb[f'incor{s}'] = fb[f'incor{s}'][ :,:num_targ_trials,: ]
            # loop over stims make big arrays that have all stims stacked...
            for s in range( n_stims ):
                # init
                if ( s==0 ):
                    sp_cor = sp[f'cor{s}']
                    h1_cor = h1[f'cor{s}']
                    h2_cor = h2[f'cor{s}']
                    ff_cor = ff[f'cor{s}']
                    fb_cor = fb[f'cor{s}']
                    sp_incor = sp[f'incor{s}']
                    h1_incor = h1[f'incor{s}']
                    h2_incor = h2[f'incor{s}']
                    ff_incor = ff[f'incor{s}']
                    fb_incor = fb[f'incor{s}']
                # else concat to stack stims on top of each other.
                else:
                    sp_cor = np.vstack( [ sp_cor,sp[f'cor{s}'] ] )
                    h1_cor = torch.hstack( [ h1_cor,h1[f'cor{s}'] ] )
                    h2_cor = torch.hstack( [ h2_cor,h2[f'cor{s}'] ] )
                    ff_cor = torch.hstack( [ ff_cor,ff[f'cor{s}'] ] )
                    fb_cor = torch.hstack( [ fb_cor,fb[f'cor{s}'] ] )
                    sp_incor = np.vstack( [ sp_incor,sp[f'incor{s}'] ] )
                    h1_incor = torch.hstack( [ h1_incor,h1[f'incor{s}'] ] )
                    h2_incor = torch.hstack( [ h2_incor,h2[f'incor{s}'] ] )
                    ff_incor = torch.hstack( [ ff_incor,ff[f'incor{s}'] ] )
                    fb_incor = torch.hstack( [ fb_incor,fb[f'incor{s}'] ] )
            # bail on the while loop
            enough = True
        #clean up...
        gc.collect()
        torch.cuda.empty_cache()
        #increment num batch counter
        nb += 1
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
    # compute mean acc
    acc = acc / nb
    #clean up...
    gc.collect()
    torch.cuda.empty_cache()
    return sp_cor,sp_incor,cl,h1_cor,h1_incor,h2_cor,h2_incor,ff_cor,ff_incor,fb_cor,fb_incor,w1,w2,ff_w,fb_w,exc1,exc2,inh1,inh2,tau1,tau2,acc

#----------------------------------------------
# do decoding - either timepoint by timepoint
# or x-time generalization on exc and inh units
#----------------------------------------------
def decode_ls_svm_exc_inh(r_mat, r_mat2, n_afc, stim_val, t_step, exc1, inh1, exc2, inh2, train_prcnt, n_cvs, tmpnt_or_cross_tmpnt, num_targ_trials):
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
        trn_ind = [ind for inds in trn_ind for ind in inds ]
        tst_ind = [ind for inds in tst_ind for ind in inds ]
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
    # return mean pred acc across all cv folds
    return np.nanmean( pred_acc_exc1,axis=0 ), np.nanmean( pred_acc_inh1,axis=0 ), np.nanmean( pred_acc_exc2,axis=0 ), np.nanmean( pred_acc_inh2,axis=0 )

def decode_ls_svm(r_mat, s_label, n_afc, w_size, time_or_xgen, trn_prcnt):
    """
    
    least square SVM
    
    Parameters
    ----------
    r_mat : matrix of rates (timepoints, trials, neurons)
    
    s_label : which stimulus was presented on this trial
        
    Returns
    -------
    over_acc: overall prediction accuracy
    stim_acc: decoding accuracy for each stimulus
        
    """
    
    # num neurons and num timepoints
    n_tmpts = r_mat.shape[0]
    n_trials = r_mat.shape[1]
    u_stims = np.unique( s_label ).astype( int )
    tmpnts = n_tmpts//w_size  # actual num of timepoints for decoding...

    
    # make the model for categorical regression - make 2 
    # to speed up x-time training...
    ls_svm_model_h = RidgeClassifier( class_weight = 'balanced' )

    
    # precalc the time windows averages
    win_data = np.array([
    np.mean(r_mat[t:t+w_size,:,:], axis=0)
    for t in range(0, n_tmpts, w_size)
    ])
    # win_data.shape = (tmpnts, n_trials, n_neurons)

    # if training/testing separately on each timepoint
    if time_or_xgen == 0: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims), tmpnts ) )
    
        # loop over timepoints 
        for t_idx in range(tmpnts):
            
            # get the data from this timepoint
            t_data = win_data[t_idx]
                      
            # train test split...because of unbalanced eval for prob 80 models
            h_train, h_test, y_train, y_test = train_test_split( t_data, s_label, train_size=trn_prcnt )        

            # train the model on data from this timepoint using train data
            ls_svm_model_h.fit( h_train,y_train )
            
            # get predictions based on test data
            pred_val = ls_svm_model_h.predict( h_test )
            
            # compute overall mean acc
            over_acc[t_idx] = np.sum( pred_val==y_test ) / len(y_test)    
            
            # Per-class accuracy
            cm = confusion_matrix(y_test, pred_val, labels=u_stims)
            stim_acc[:, t_idx] = np.diag(cm) / cm.sum(axis=1)  
        

    # generalize across time...
    elif time_or_xgen == 1: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( tmpnts,tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),tmpnts,tmpnts ) )
         
        
        # outer loop over timepoints 
        for t_idx,t in enumerate( range( 0,n_tmpts,w_size ) ) :

            # train test split
            train_idx, test_idx = train_test_split(
                np.arange(n_trials), train_size=trn_prcnt,
                stratify=s_label, random_state=1)
            
            
            h_train = win_data[t_idx][train_idx]
            y_train = s_label[train_idx]
        
            # train h models
            ls_svm_model_h.fit( h_train, y_train )

            # inner loop over timepoints 
            for tt_idx,tt in enumerate( range( 0,n_tmpts,w_size ) ) :           
                
               # test data for this timepoint
               tst_t_data = win_data[tt_idx][test_idx]
               tst_t_y = s_label[test_idx]              
           
               # get predictions based on test data
               pred_val = ls_svm_model_h.predict( tst_t_data )
               
               # compute overall mean acc
               over_acc[t_idx,tt_idx] = np.sum( pred_val==tst_t_y ) / len(tst_t_y)    
               
               # Per-class accuracy
               cm = confusion_matrix(tst_t_y, pred_val, labels=u_stims)
               stim_acc[:, t_idx, tt_idx] = np.diag(cm) / cm.sum(axis=1)  
                

    return over_acc, stim_acc


def decode_svc(stim_prob, r_mat, s_label, trn_prcnt, time_or_xgen, w_size, num_cs, n_cvs_for_grid, max_iter):
    """
    
    linear support vector classifer
    
    Parameters
    ----------
    r_mat : matrix of rates (timepoints, trials, units)
    
    s_label : which stimulus was presented on this trial
        
    trn_prcnt : percent of data to use as training set
    
    time_or_xgen : timepoint by timepoint or xgen over time
    
    w_size : time window size to avg the data before decoding - will step across
        time in steps of this size...
        
    n_cs : number of Cs to eval in training set before testing
    
    n_cvs_for_grid : number of cv folds to use to esimate best C 
        based on training data
        
    max_iter : max fitting iterations for convergence
    
    Returns
    -------
    mean_acc : overall prediction accuracy
    stim_acc : decoding accuracy for each of the N possible stimuli
        
    """
    
    # build the model...just searching over C for now...
    if num_cs > 1:
        Cs = np.logspace( -10,0,num_cs )
    else:
        Cs = [1]
        
    # Define parameter grid
    param_grid = {
        'C': Cs
        # 'loss': ['hinge', 'squared_hinge'],
        # 'penalty': ['l2']
    }
     
    # set up grid search model object
    model = GridSearchCV( estimator=LinearSVC( class_weight='balanced', max_iter=max_iter ),
                               param_grid=param_grid,
                               cv=n_cvs_for_grid,
                               scoring='accuracy' )
    
    # basic info
    n_tmpts = r_mat.shape[0]
    u_stims = np.unique( s_label ).astype( int )
    tmpnts = n_tmpts//w_size  # actual num of timepoints for decoding...
    

    # blance classes if needed
    if stim_prob == 0.8:
        # classes and their counts
        classes, class_counts = np.unique(s_label, return_counts = True)
        max_class = classes[np.argmax(class_counts)] # how many expected stim
        min_class_size = np.min(class_counts[class_counts != np.max(class_counts)]) # what's the smallest num unexpected stim
        
        # # Collect balanced indices
        balanced_indices = []
        
        for cls in classes:
            cls_indices = np.where(s_label == cls)[0]
            if cls == max_class:
                # Undersample dominant class
                selected = np.random.choice(cls_indices, min_class_size, replace=False)
            else:
                # Keep all samples from minority classes
                selected = cls_indices
            balanced_indices.append(selected)
        
        # Concatenate and shuffle
        balanced_indices = np.concatenate(balanced_indices)
        np.random.seed(42)
        np.random.shuffle(balanced_indices)
        
        # Apply to data and labels
        r_mat = r_mat[:,balanced_indices,:]
        s_label = s_label[balanced_indices]
    
    
    # precalc the time windows averages
    win_data = np.array([
    np.mean(r_mat[t:t+w_size,:,:], axis=0)
    for t in range(0, n_tmpts, w_size)
    ])
    # win_data.shape = (tmpnts, n_trials, n_neurons)
    
    
    # if training/testing separately on each timepoint
    if time_or_xgen == 0: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),tmpnts ) )
    
        # loop over timepoints 
        for t_idx in range(tmpnts):
            
            # get the data from this timepoint
            t_data = win_data[t_idx]
                                       
            # train test split...because of unbalanced eval for prob 80 models
            X_trn, X_tst, y_trn, y_tst = train_test_split( t_data, s_label, train_size=trn_prcnt )        

            # train the model on data from this timepoint using train data
            model.fit( X_trn,y_trn )
            
            # get predictions based on test data
            pred_val = model.predict( X_tst )
            
            # Overall accuracy
            over_acc[t_idx] = np.sum( pred_val==y_tst ) / len(y_tst)    
            
            # Per-class accuracy
            cm = confusion_matrix(y_tst, pred_val, labels=u_stims)
            stim_acc[:, t_idx] = np.diag(cm) / cm.sum(axis=1)

            
            
    # train on one timepoint, gen to all others, etc...
    elif time_or_xgen == 1: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( tmpnts,tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),tmpnts,tmpnts ) )
      
        # outer loop over timepoints 
        for t_idx in range(tmpnts):
            
            # get a train/test index...
            split_ind = train_test_split(np.arange(len(s_label)), train_size=trn_prcnt)
            
            # get the train data from this timepoint
            trn_t_data = win_data[t_idx][split_ind[0], :]
            trn_t_y = s_label[ split_ind[0] ]
            

            # train the model on data from this timepoint using train data
            model.fit( trn_t_data,trn_t_y )
            
            # inner loop over timepoints to generalize the trained model
            for tt_idx,tt in enumerate( range( 0,n_tmpts,w_size ) ):   
            
                # test data for this timepoint
                tst_t_data = np.mean( r_mat[tt:tt+w_size,split_ind[1],:],axis=0 )
                tst_t_y = s_label[ split_ind[1] ]                    
            
                # get predictions based on test data
                pred_val = model.predict( tst_t_data )
                
                # Overall accuracy
                over_acc[t_idx,tt_idx] = np.sum( pred_val==tst_t_y ) / len(tst_t_y)    
                
                # Per-class accuracy
                cm = confusion_matrix(tst_t_y, pred_val, labels=u_stims)
                stim_acc[:, t_idx, tt_idx] = np.diag(cm) / cm.sum(axis=1)                

    # return decoding accuracy
    return over_acc,stim_acc



def confusion_matrix_reproduction( outputs,s_label,settings ):  
        
    # range of target
    t_onset = settings['stim_on'] + settings['stim_dur']
    

    y_predicted = np.zeros( settings['batch_size'] )
    y_actual = s_label.astype(int)
    
    
    # loop over number of trials
    for nt in range( settings['batch_size'] ):
        
    
        idx = np.where( np.mean(outputs[t_onset:, nt, :], axis = 0) > settings['acc_amp_thresh'] )[0]
        
        if len(idx)>0:
            y_predicted[nt] = int(idx[0])
        else:
            y_predicted[nt] = -1 # no channel was marked as stim
            
    valid_labels = np.append(np.unique(y_actual), -1)
    above_thresh_trials = np.where(y_predicted != -1)
    sub_thresh_trials = np.where(y_predicted == -1)
    cm = confusion_matrix(y_actual, y_predicted, labels = valid_labels, normalize = 'true')    
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = valid_labels)
    # disp.plot(cmap='Blues')  # 'd' for integers
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.show()
    
    # which stimuli does the response not reach threshold?
    sub_thresh_labels = np.unique(s_label[sub_thresh_trials[0][:]])
    
            
        
    # compute mean acc over all trials in the current batch and return
    return cm, sub_thresh_labels




# TODO
def confusion_matrix_rdk( outputs,s_label,settings ):  
        
    # range of target
    t_onset = settings['stim_on'] + settings['stim_dur']
    

    y_predicted = np.zeros( settings['batch_size'] )
    y_actual = np.zeros( settings['batch_size'] )
    
    
    # loop over number of trials
    for nt in range( settings['batch_size'] ):
        
        if s_label[nt] == 0:
            y_actual[nt] = 0
        else:
            y_actual[nt] = 1
    
        # figure out if this trial is 'correct' based on acc_amp_thresh
        if ( s_label[nt] == 0 ):
            if np.mean(outputs[t_onset:, nt, :], axis = 0) > settings['acc_amp_thresh']:
                y_predicted[nt] = 0
            elif np.mean(outputs[t_onset:, nt, :], axis = 0) < ( 1 - settings['acc_amp_thresh'] ):
                y_predicted[nt] = 1
            else:
                y_predicted[nt] = -1

        else:
            if np.max( np.abs( np.mean(outputs[t_onset:, nt, :], axis = 0 ) ) < ( 1 - settings['acc_amp_thresh'] ) ):
                y_predicted[nt] = 1
            elif np.mean(outputs[t_onset:, nt, :], axis = 0) > settings['acc_amp_thresh']:
                y_predicted[nt] = 0
            else:
                y_predicted[nt] = -1

            
    valid_labels = np.append(np.unique(y_actual), -1)
    above_thresh_trials = np.where(y_predicted != -1)
    sub_thresh_trials = np.where(y_predicted == -1)
    cm = confusion_matrix(y_actual, y_predicted, labels = valid_labels, normalize = 'true')    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = valid_labels)
    disp.plot(cmap='Blues')  # 'd' for integers
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # which stimuli does the response not reach threshold?
    sub_thresh_labels = np.unique(s_label[sub_thresh_trials[0][:]])
    
            
        
    # compute mean acc over all trials in the current batch and return
    return cm, sub_thresh_labels
