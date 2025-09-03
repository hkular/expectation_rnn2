#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:05:11 2025

@author: johnserences, jserences@ucsd.edu
"""

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt   # for debugging...

#---------------------------------------------------------------
# define rdk task...
#---------------------------------------------------------------
class RDKtask:
    
    """
    Firing-rate RNN model for excitatory and inhibitory neurons
    Initialization of the firing-rate model with recurrent connections
    """
    def __init__(self, settings):

        # get init params from task dictionary
        self.task = settings['task']

        if ( self.task == 'rdk' ) | ( self.task == 'rdk_reproduction' ) | ( self.task == 'rdk_repro_cue' ) :

            # if passes check, assign
            self.n_afc = settings['n_afc']       # number of stimulus alternatives
            self.T = settings['T']                   # timepoints in each trial
            self.stim_on = settings['stim_on']       # stimulus onset time
            self.stim_dur = settings['stim_dur']     # stimulus duration
            self.stim_prob = settings['stim_prob']   # probability of stim 1, with probability of (1-stim_prob))/(n_afc-1) for all other options
            self.stim_amp = settings['stim_amp']     # amplitude of stimulus during training
            self.stim_noise = settings['stim_noise'] # amp of background randn noise in the stim
            self.batch_size = settings['batch_size'] # number of trials in each training batch
            self.acc_amp_thresh = settings['acc_amp_thresh']  # threshold on output channels for determining accuracy
            self.out_size = settings['out_size']
            
        else:
            raise ValueError(f'{self.task} is not a supported task')            

        # grab a few additional params for the cue task
        if ( self.task == 'rdk_repro_cue' ):
            self.num_cues = settings['num_cues']
            self.cue_on = settings['cue_on']
            self.cue_dur = settings['cue_dur']
        self.rand_seed_bool = settings['rand_seed_bool']    
        if settings['rand_seed_bool']:   
            self.seed = 42

    #---------------------------------------------------------------
    # rdk task
    #---------------------------------------------------------------
    def generate_rdk_stim(self):
        """
        Generate the [time x trials in batch] input stimulus matrix 
        for the "rdk" motion discrimination task
    
        INPUT
            settings: dict containing the following keys
                n_afc: how many stim alternatives
                T: duration of a single trial (in steps)
                stim_on: stimulus starting time (in steps)
                stim_dur: stimulus duration (in steps)
                stim_prob: probability of stim 1 ( and all other options are (1-stim_prob))/(n_afc-1) ) 
                stim_amp: amp of stimulus (to mimic changing motion coherence)
                batch_size: number of trials to generate for each batch
                
        OUTPUT
            u: T x batch_size x n_afc matrix
            s_label: batch_size vector labeling stim on each trial
        
        """
        
        # alloc storage for stim time series 'u' and stim identity on each trial
        # in this batch...u is a tensor, but stim_pres is ok as np array
        #u = torch.zeros( ( self.T,self.batch_size,self.n_afc ) ) 
        if self.rand_seed_bool:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
                  
        # rand init...
        u = torch.randn( ( self.T,self.batch_size,self.n_afc ) ) * self.stim_noise
        
        s_label = np.full( self.batch_size,np.nan )        
        
        # generate all trials in this batch
        for nt in range( self.batch_size ):    

            # pick a stim based on probability (expectation)
            # and scale stim based on desired amp...
            
            if np.random.rand() <= self.stim_prob:   # decide if its gunna be biased stim
                # always channel 0...
                u[ self.stim_on:(self.stim_on+self.stim_dur),nt,0 ] += self.stim_amp
                s_label[nt] = 0
            
            # randomly pick one of the other n_afc-1 stims
            else:
                # random int between 1 and n_afc (channel 0 always reserved for the biased stim)
                ind = np.random.choice( np.arange( 1, self.n_afc ) )
            
                # assign the stim to the selected channel
                u[ self.stim_on:(self.stim_on+self.stim_dur),nt,ind ] += self.stim_amp
                s_label[nt] = ind
                     
        return u, s_label
    
    #---------------------------------------------------------------
    # target timeseries 
    #---------------------------------------------------------------
    def generate_rdk_target(self, s_label):
        
        '''
        Generate the [time x trial in batch] target matrix 
    
        INPUT
            
            T: duration of a single trial (in steps)
            batch_size: number of trials to generate for each batch
            stim_pres: matrix indicating which stim was presented on each trial
                
        OUTPUT
            targs: [T x batch_size] target matrix, type torch.Tensor (float)
            
        '''
        
        # define target outputs for model training
        targs = torch.zeros( ( self.T, self.batch_size ) )
        t_onset = self.stim_on+self.stim_dur
        
        # loop over trials in this batch
        for nt in range( self.batch_size ):
            # if there was an stim in the 'biased' channel (0)
            if s_label[ nt ] == 0:
                targs[ t_onset:,nt ] = 1
            # else:
            #     if self.output_type == 1:
            #         targs[ t_onset:,nt ] = -1
                
        return targs
    
    #---------------------------------------------------------------
    # rdk task - reproduction
    #---------------------------------------------------------------
    def generate_rdk_reproduction_stim(self):
        """
        Generate the [time x trials in batch] input stimulus matrix 
        for the "rdk" motion discrimination task
    
        INPUT
            settings: dict containing the following keys
                n_afc: how many stim alternatives
                T: duration of a single trial (in steps)
                stim_on: stimulus starting time (in steps)
                stim_dur: stimulus duration (in steps)
                stim_prob: probability of stim 1 ( and all other options are (1-stim_prob))/(n_afc-1) ) 
                stim_amp: amp of stimulus (to mimic changing motion coherence)
                batch_size: number of trials to generate for each batch
                
        OUTPUT
            u: T x batch_size x n_afc matrix
            s_label: batch_size vector labeling stim on each trial
        
        """
        
        # alloc storage for stim time series 'u' and stim identity on each trial
        # in this batch...u is a tensor, but stim_pres is ok as np array
        #u = torch.zeros( ( self.T,self.batch_size,self.n_afc ) ) 
        if self.rand_seed_bool:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # rand init...
        u = torch.randn( ( self.T,self.batch_size,self.n_afc ) ) * self.stim_noise
        
        s_label = np.full( self.batch_size,np.nan )        
        
        # generate all trials in this batch
        for nt in range( self.batch_size ):    

            # pick a stim based on probability (expectation)
            # and scale stim based on desired amp...
            
            if np.random.rand() <= self.stim_prob:   # decide if its gunna be biased stim
                # always channel 0...
                u[ self.stim_on:(self.stim_on+self.stim_dur),nt,0 ] += self.stim_amp
                s_label[nt] = 0
            
            # randomly pick one of the other n_afc-1 stims
            else:
                # random int between 1 and n_afc (channel 0 always reserved for the biased stim)
                ind = np.random.choice( np.arange( 1, self.n_afc ) )
            
                # assign the stim to the selected channel
                u[ self.stim_on:(self.stim_on+self.stim_dur),nt,ind ] += self.stim_amp
                s_label[nt] = ind
                     
        return u, s_label
    
    #---------------------------------------------------------------
    # target timeseries for basic reproduction task
    #---------------------------------------------------------------
    def generate_rdk_reproduction_target(self, s_label):
        
        '''
        Generate the [time x trial in batch] target matrix 
    
        INPUT
            
            T: duration of a single trial (in steps)
            batch_size: number of trials to generate for each batch
            stim_pres: matrix indicating which stim was presented on each trial
                
        OUTPUT
            targs: [T x batch_size] target matrix, type torch.Tensor (float)
            
        '''
        
        # define target outputs for model training
        targs = torch.zeros( ( self.T, self.batch_size, self.n_afc ) )
        t_onset = self.stim_on+self.stim_dur
        stim_labels = s_label.astype( int )
        
        # loop over trials in this batch
        for nt in range( self.batch_size ):
            
            # put a 1 in the right channel
            targs[ t_onset:,nt,stim_labels[nt] ] = 1

        return targs
    
    #---------------------------------------------------------------
    # rdk task - reproduction w cue
    #---------------------------------------------------------------
    def generate_rdk_reproduction_cue_stim(self):
        """
        Generate the [time x trials in batch] input stimulus matrix 
        for the "rdk" motion discrimination task
    
        INPUT
            settings: dict containing the following keys
                n_afc: how many stim alternatives
                T: duration of a single trial (in steps)
                stim_on: stimulus starting time (in steps)
                stim_dur: stimulus duration (in steps)
                stim_prob: probability of stim 1 ( and all other options are (1-stim_prob))/(n_afc-1) ) 
                stim_amp: amp of stimulus (to mimic changing motion coherence)
                batch_size: number of trials to generate for each batch
                
        OUTPUT
            u: T x batch_size x n_afc matrix
            s_label: batch_size vector labeling stim on each trial
            c_label: T x batch_size x n_afc matrix
        
        """
        
        # alloc storage for stim time series 'u' and stim identity on each trial
        # in this batch...u is a tensor, but stim_pres is ok as np array
        #u = torch.zeros( ( self.T,self.batch_size,self.n_afc ) ) 
        if self.rand_seed_bool:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # rand init...
        u = torch.randn( ( self.T,self.batch_size,self.n_afc ) ) * self.stim_noise  # stim
        s_label = np.zeros( ( self.batch_size,self.n_afc ) )
        c = torch.zeros( ( self.T,self.batch_size,self.num_cues ) )                    # cue 
        c_label = np.zeros( self.batch_size )
      
        
        # generate all trials in this batch
        for nt in range( self.batch_size ):    

            # make the s->r mapping cue
            ind = np.random.choice( np.arange( 0, self.num_cues ) )
            c[ self.cue_on:(self.cue_on+self.cue_dur),nt,ind ] = 1              
            c_label[ nt ] = ind
            

            # pick a stim based on probability (expectation)
            # and scale stim based on desired amp...
            if np.random.rand() <= self.stim_prob:   # decide if its gunna be biased stim
                # always channel 0...
                u[ self.stim_on:(self.stim_on+self.stim_dur),nt,0 ] += self.stim_amp
                s_label[nt,0] = 1

            # randomly pick one of the other n_afc-1 stims
            else:
                # random int between 1 and n_afc (channel 0 always reserved for the biased stim)
                ind = np.random.choice( np.arange( 1, self.n_afc ) )
            
                # assign the stim to the selected channel
                u[ self.stim_on:(self.stim_on+self.stim_dur),nt,ind ] += self.stim_amp
                s_label[nt,ind] = 1
                
                     
        return u, c, s_label, c_label.astype(int)
    
    #---------------------------------------------------------------
    # generate the scrambled sr-mapping for this model
    #---------------------------------------------------------------    
    def gen_sr_scram( self ):
        
        if self.rand_seed_bool:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            
        # for randomly scrambling the s->r mapping
        sr_scram = np.full( ( self.num_cues,self.n_afc ),np.nan )
        check_repeat = True
        while check_repeat:
            repeat = False
            for sr in range( self.num_cues ):
                sr_scram[ sr,: ] = np.random.permutation( np.arange( self.n_afc ) )
            
                for sr2 in range( self.num_cues ):
                    if sr != sr2:
                        if np.all( sr_scram[sr,:] == sr_scram[sr2,:] ):
                            repeat = True
                    
            if repeat == False:
                check_repeat = False
    
        return sr_scram.astype( int )
    
    #---------------------------------------------------------------
    # target timeseries 
    #---------------------------------------------------------------
    def generate_rdk_reproduction_cue_target(self, s_label, sr_scram, c_label):
        
        '''
        Generate the [time x trial in batch] target matrix 
    
        INPUT
            
            T: duration of a single trial (in steps)
            batch_size: number of trials to generate for each batch
            stim_pres: matrix indicating which stim was presented on each trial
                
        OUTPUT
            targs: [T x batch_size] target matrix, type torch.Tensor (float)
            
        '''
        
        # define target outputs for model training
        targs = torch.zeros( ( self.T, self.batch_size, self.n_afc ) )
        t_onset = self.stim_on+self.stim_dur
        stim_labels = s_label.astype( int )

        
        
        # loop over trials in this batch
        for nt in range( self.batch_size ):
            
            # loop over stim channels
            for sc in range( self.n_afc ):
                
                # if there was a stim in this channel, then 
                # generate target output
                if s_label[ nt,sc ] != 0:
                    
                    targs[ t_onset:,nt,sr_scram[ c_label[ nt ],sc ] ] = s_label[nt,sc]


        return targs

    
    #---------------------------------------------------------------
    # Compute task accuracy using defined criteria
    #---------------------------------------------------------------
    def compute_acc( self,outputs,s_label ): 
            
        # range of target
        t_onset = self.stim_on+self.stim_dur
        
        # compute mean of model output on each trial in last batch over 
        # last timepoints in each trial
        tmp_acc = np.zeros( self.batch_size )
        
        # loop over number of trials
        for nt in range( self.batch_size ):
            
            # get the values of the output in the channel corresponding to the stim
            # on each trial
            mean_out = torch.mean( torch.squeeze( outputs[ t_onset:,nt,: ] ),axis=0 )            
            
            # figure out if this trial is 'correct' based on acc_amp_thresh
            if ( s_label[nt] == 0 ):
                if ( mean_out > self.acc_amp_thresh ):
                    tmp_acc[ nt ] = 1
            else:
                if ( torch.max( torch.abs( mean_out ) ) < ( 1 - self.acc_amp_thresh ) ):
                    tmp_acc[ nt ] = 1                
                 
        # compute mean acc over all trials in the current batch and return
        return np.mean(tmp_acc), tmp_acc
    
    #---------------------------------------------------------------
    # Compute task accuracy using defined criteria
    #---------------------------------------------------------------
    def compute_acc_reproduction( self,outputs,s_label ): 
            
        # range of target
        t_onset = self.stim_on+self.stim_dur
        
        # compute mean of model output on each trial in last batch over 
        # last timepoints in each trial
        tmp_acc = np.zeros( self.batch_size )
        stim_label = s_label.astype( int )
        
        # loop over number of trials
        for nt in range( self.batch_size ):
            
            # get the values of the output in the channel corresponding to the stim
            # on each trial
            mean_out_targ = torch.mean( outputs[ t_onset:,nt,stim_label[nt] ],axis=0 ) 
            nt_ind = np.setdiff1d( np.arange( self.n_afc ),stim_label[nt] )
            mean_out_non_targ = torch.mean( torch.mean( outputs[ t_onset:,nt,nt_ind ],axis=0 ) )           

            # figure out if this trial is 'correct' based on acc_amp_thresh being high 
            # in the target channel AND low in the non-target channcels
            if ( mean_out_targ > self.acc_amp_thresh ) & ( mean_out_non_targ < (1 - self.acc_amp_thresh ) ):
                tmp_acc[ nt ] = 1
            
        # compute mean acc over all trials in the current batch and return
        return np.mean(tmp_acc), tmp_acc
    
    #---------------------------------------------------------------
    # Compute task accuracy for repro cue task
    #---------------------------------------------------------------
    def compute_acc_reproduction_cue( self,outputs,s_label,targets ): 
            
        # range of target
        t_onset = self.stim_on+self.stim_dur
        
        # preallocate accuracy container and get targets
        tmp_acc = np.zeros( self.batch_size )
        targs = targets.cpu()
        
        # loop over number of trials
        for nt in range( self.batch_size ):
            
            # get the values of the output in the channel corresponding to the right 
            # target channel on each trial and non-target responses
            t_ind = np.where(targs[-1,nt,:]==1)[0]
            nt_ind = np.setdiff1d( np.arange( self.n_afc ),t_ind )
            
            # compute mean of model output on each trial in last batch over 
            # last timepoints in each trial
            mean_out_targ = torch.mean( outputs[ t_onset:,nt,t_ind ],axis=0 ) 
            mean_out_non_targ = torch.mean( torch.mean( outputs[ t_onset:,nt,nt_ind ],axis=0 ) )           

            # figure out if this trial is 'correct' based on acc_amp_thresh being high 
            # in the target channel AND low in the non-target channcels
            if ( mean_out_targ > self.acc_amp_thresh ) & ( mean_out_non_targ < (1 - self.acc_amp_thresh ) ):
                tmp_acc[ nt ] = 1
            
        # compute mean acc over all trials in the current batch and return
        return np.mean(tmp_acc), tmp_acc
    
    