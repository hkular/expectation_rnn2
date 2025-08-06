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

# load the model, turn off grads
def load_model( fn,device ): 
    
    # load model...
    model = torch.load(fn,map_location=torch.device(device))

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

    return outputs,s_label,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau1,tau3,m_acc,tbt_acc,cues


#stim_prob, r_mat, s_label, trn_prcnt, n_trn_tst_cvs, time_or_xgen, w_size, num_cs, n_cvs_for_grid, max_iter
def decode_ls_svm(r_mat, s_label, n_afc, w_size, time_or_xgen, n_trn_tst_cvs, trn_prcnt):
    """
    
    least square SVM
    
    Parameters
    ----------
    r_mat : matrix of rates (timepoints, neurons)
    
    s_label : which stimulus was presented on this trial
        
    Returns
    -------
    pred acc for layer that was given as an input
        
    """
    
    # num neurons and num timepoints
    n_tmpts = r_mat.shape[0]
    n_trials = r_mat.shape[1]
    n_neurons = r_mat.shape[2]
    u_stims = np.unique( s_label ).astype( int )
    tmpnts = n_tmpts//w_size  # actual num of timepoints for decoding...
    
    # make the model for categorical regression - make 2 
    # to speed up x-time training...
    # svc_model = LinearSVC( class_weight = 'balanced', max_iter=1000 )
    # svc_model2 = LinearSVC( class_weight = 'balanced', max_iter=1000 )
    ls_svm_model_h = RidgeClassifier( class_weight = 'balanced' )

    
    # if training/testing separately on each timepoint
    if time_or_xgen == 0: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( n_trn_tst_cvs,tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),n_trn_tst_cvs,tmpnts ) )
    
        # loop over timepoints 
        for t_idx,t in enumerate( range( 0,n_tmpts,w_size ) ) :
            
            t_data = np.mean( r_mat[t:t+w_size,:,:],axis=0 )
            
            # cv loop 
            for cv in range( n_trn_tst_cvs ):
                

                # train test split...because of unbalanced eval for prob 80 models
                h_train, h_test, y_train, y_test = train_test_split( t_data, s_label, train_size=trn_prcnt )        

                # train the model on data from this timepoint using train data
                ls_svm_model_h.fit( h_train,y_train )
                
                # get predictions based on test data
                pred_val = ls_svm_model_h.predict( h_test )
                
                # compute overall mean acc
                over_acc[cv,t_idx] = np.sum( pred_val==y_test ) / len(y_test)    
                
                #then compute acc for each stim at this timepoint
                for us in u_stims:
                    # index of trials in test set that are the
                    # current stim (us)
                    s_idx = np.where( y_test==us )[0]
                    stim_acc[ us,cv,t_idx ] = np.sum( pred_val[ s_idx ]==us ) / len(s_idx)    
        
        # then avg accs over cv folds before returning
        over_acc = np.mean( over_acc,axis=0 )
        stim_acc = np.mean( stim_acc,axis=1 )

    # generalize across time...
    elif time_or_xgen == 1: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( n_trn_tst_cvs,tmpnts,tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),n_trn_tst_cvs,tmpnts,tmpnts ) )
         
        
        # outer loop over timepoints 
        for t_idx,t in enumerate( range( 0,n_tmpts,t_step ) ) :

            # train test split
            rnd_tri = np.random.permutation( n_trials )
            
            h_train = r_mat[t,rnd_tri[:trial_split],:]
            y_train = s_label[rnd_tri[:trial_split]]

            
            # train h models
            ls_svm_model_h.fit( h_train, y_train )

            # inner loop over timepoints 
            for tt_idx,tt in enumerate( range( 0,n_tmpts,t_step ) ) :           
                
                # compute acc for h
                pred_acc[ t_idx,tt_idx ] = ls_svm_model_h.score( r_mat[tt,rnd_tri[trial_split:],:],s_label[rnd_tri[trial_split:]] )
                
                
                
                

    return pred_acc


def decode_svc(stim_prob, r_mat, s_label, trn_prcnt, n_trn_tst_cvs, time_or_xgen, w_size, num_cs, n_cvs_for_grid, max_iter):
    """
    
    linear support vector classifer
    
    Parameters
    ----------
    r_mat : matrix of rates (timepoints, trials, units)
    
    s_label : which stimulus was presented on this trial
        
    trn_prcnt : percent of data to use as training set
    
    n_trn_tst_cvs : number of train/test cv folds 
    
    time_or_xgen : timepoint by timepoint or xgen over time
    
    w_size : time window size to avg the data before decoding - will step across
        time in steps of this size...
        
    n_cs : number of Cs to eval in training set before testing
    
    n_cvs_for_grid : number of cv folds to use to esimate best C 
        based on training data
        
    max_iter : max fitting iterations for convergence
    
    Returns
    -------
    pred_val : np array, predicted stim on each trial
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
    n_trials = r_mat.shape[1]
    n_neurons = r_mat.shape[2]
    u_stims = np.unique( s_label ).astype( int )
    tmpnts = n_tmpts//w_size  # actual num of timepoints for decoding...
    
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
    
    
    # if training/testing separately on each timepoint
    if time_or_xgen == 0: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( n_trn_tst_cvs,tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),n_trn_tst_cvs,tmpnts ) )
    
        # loop over timepoints 
        for t_idx,t in enumerate( range( 0,n_tmpts,w_size ) ):
            
            # get the data from this timepoint so we don't
            # have to keep re-indexing in the cv loop
            # mean across time window
            t_data = np.mean( r_mat[t:t+w_size,:,:],axis=0 )
                             
            
            # cv loop 
            for cv in range( n_trn_tst_cvs ):

                # train test split...because of unbalanced eval for prob 80 models
                X_trn, X_tst, y_trn, y_tst = train_test_split( t_data, s_label, train_size=trn_prcnt )        

                # train the model on data from this timepoint using train data
                model.fit( X_trn,y_trn )
                
                # get predictions based on test data
                pred_val = model.predict( X_tst )
                
                # compute overall mean acc
                over_acc[cv,t_idx] = np.sum( pred_val==y_tst ) / len(y_tst)    
                
                #then compute acc for each stim at this timepoint
                for us in u_stims:
                    # index of trials in test set that are the
                    # current stim (us)
                    s_idx = np.where( y_tst==us )[0]
                    stim_acc[ us,cv,t_idx ] = np.sum( pred_val[ s_idx ]==us ) / len(s_idx)    
        
        # then avg accs over cv folds before returning
        over_acc = np.mean( over_acc,axis=0 )
        stim_acc = np.mean( stim_acc,axis=1 )
            
            
    # train on one timepoint, gen to all others, etc...
    elif time_or_xgen == 1: 
    
        # alloc to store the overall acc and the stim-specific acc
        over_acc = np.zeros( ( n_trn_tst_cvs,tmpnts,tmpnts ) )  
        stim_acc = np.zeros( ( len(u_stims),n_trn_tst_cvs,tmpnts,tmpnts ) )
      
        # outer loop over timepoints 
        for t_idx,t in enumerate( range( 0,n_tmpts,w_size ) ):
            
            # get a train/test index...
            split_ind = train_test_split(np.arange(len(s_label)), train_size=trn_prcnt, shuffle=False)
            
            # get the train data from this timepoint
            trn_t_data = np.mean( r_mat[t:t+w_size,split_ind[0],:],axis=0 )
            trn_t_y = s_label[ split_ind[0] ]
            
            # cv loop 
            for cv in range( n_trn_tst_cvs ):

                # train the model on data from this timepoint using train data
                model.fit( trn_t_data,trn_t_y )
                
                # inner loop over timepoints to generalize the trained model
                for tt_idx,tt in enumerate( range( 0,n_tmpts,w_size ) ):   
                
                    # test data for this timepoint
                    tst_t_data = np.mean( r_mat[tt:tt+w_size,split_ind[1],:],axis=0 )
                    tst_t_y = s_label[ split_ind[1] ]                    
                
                    # get predictions based on test data
                    pred_val = model.predict( tst_t_data )
                    
                    # compute overall mean acc
                    over_acc[cv,t_idx,tt_idx] = np.sum( pred_val==tst_t_y ) / len(tst_t_y)    
                    
                    #then compute acc for each stim at this timepoint
                    for us in u_stims:
                        # index of trials in test set that are the
                        # current stim (us)
                        s_idx = np.where( tst_t_y==us )[0]
                        stim_acc[ us,cv,t_idx,tt_idx ] = np.sum( pred_val[ s_idx ]==us ) / len(s_idx)             
            
        # then avg accs over cv folds before returning
        over_acc = np.mean( over_acc,axis=0 )
        stim_acc = np.mean( stim_acc,axis=1 )        

    # return over 
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
