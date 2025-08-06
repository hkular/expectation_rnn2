#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:46:51 2025

@author: johnserences, jserences@ucsd.edu

Main class for training/evaluating the model...
"""

import torch
import numpy as np
from inp_layer import *
from cue_layer import *
from h_layer import *
import matplotlib.pyplot as plt   # for debugging

#--------------------------------
# class to define the network and to regulate
# information FF and FB flow through network 
#--------------------------------
class RNN(torch.nn.Module):

    #----------------------------
    # set up the instance...
    #----------------------------
    def __init__(self, rnn_settings):
        '''
        Initialize the params for the network
            INPUT
                rnn_settings: dict with network parameters
        '''
        
        # inherit
        super().__init__()
        
        #----------------------------
        # get input params for network
        #----------------------------
        self.cue_layer_num = rnn_settings['cue_layer_num']  # which layer is the cue?
        self.task_type = rnn_settings['task_type']          # which task
        self.dt = rnn_settings['dt']                        # timestep
        self.batch_size = rnn_settings['batch_size']        # number of trials in a training batch
        self.stim_on = rnn_settings['stim_on']              # onset time of stim - only apply feedback after this time
        self.inp_size = rnn_settings['inp_size']            # number of inputs, last dim of a [time x batch size x num_inputs] matrix (where batch size == n_trials) 
        self.out_size = rnn_settings['out_size']            # output size - defaults to inp_size
        self.noise = rnn_settings['noise']                  # noise before passing x_t to activation function
        self.apply_dale = rnn_settings['apply_dale']        # using dale's principle? 
        act_func = rnn_settings['act_func']                 # activation function relating x_t to r
        if act_func == 'relu':
            self.act_func = torch.relu
        elif act_func == 'sigmoid':
            self.act_func = torch.sigmoid
        elif act_func == 'tanh':
            self.act_func = torch.tanh
        else:
            raise ValueError(f'{act_func} is not a currently supported activation function')

        # create the input layer, input size x ring layer size
        self.inp_layer = InpLayer( rnn_settings )

        # if doing the cueing task...
        if self.task_type == 'rdk_repro_cue':
            self.cue_layer = CueLayer( rnn_settings )

        # then build a dictionary of params for each 
        # hidden layer...just the stuff that we'll need
        # for the recurrence computations, not all params
        # governing the initialization of weights, etc (that
        # will happen in the h_layer module)        
        self.n_h_layers = rnn_settings['n_h_layers']   # xx don't really need in the hardcoded version...
        self.h_size1 = rnn_settings['h_size'][0]       # size of first layer, etc
        self.h_size2 = rnn_settings['h_size'][1]
        self.h_size3 = rnn_settings['h_size'][2] 

        #--------------------------------
        # set up layer 1 and feed forward (ff) 
        # matrices for 1 to 2
        #--------------------------------
        self.h_tau1 = rnn_settings['h_tau'][0]   # taus for this layer
        
        # actually create this hidden layer, specifying the number 
        # of the layer that we want so we know what params to use from 
        # rnn_settings
        self.h_layer1 = HiddenLayer( rnn_settings,0 ) 
        
        # generate distribution of taus - will be run through act func
        # and scaled when using...
        self.h_layer1.h_taus_gaus = torch.nn.Parameter( torch.randn( rnn_settings['h_size'][0] ), requires_grad = rnn_settings['h_tau_trainable'][0] )
        
        # build a weight matrix to project activity from the 1st layer to 
        # the second layer...only do for the 0:(n-1) layers because the 
        # last hidden layer will project to the output layer
        # NOTE: I'm using a "from-to" naming convention so ff_12 is ff from layer 1 to 2...
        w = torch.zeros( ( self.h_size1,self.h_size2 ), dtype = torch.float32 ) 
        ind = torch.where( torch.rand( self.h_size1,self.h_size2 ) < rnn_settings['p_rec'][0] )
        w[ ind[0], ind[1] ]  = torch.normal( torch.zeros( len(ind[0]) ), torch.ones( len(ind[0] ) ) ) * rnn_settings['W_ff_scalar'][0]
        # if applying Dale's law, then abs ff weights
        if self.apply_dale[0] == True:
            w = torch.abs(w)
        # assign as a parameter of the first layer
        self.h_layer1.wff_12 = torch.nn.Parameter( w ,requires_grad = rnn_settings['W_ff_trainable'][0] )
        
        # mask so that only ff from exc units passes...this is the same
        # as the mask used to enforce Dale's law, but with the -1s converted
        # to zero (use relu to get rid of neg values)
        self.h_layer1.mff_12 = torch.relu( self.h_layer1.mask.clone() )
    
        #--------------------------------
        # then make 2nd layer - also provides 
        # feedback to first layer
        #--------------------------------
        self.h_tau2 = rnn_settings['h_tau'][1]   
        self.h_layer2 = HiddenLayer( rnn_settings,1 )
        self.h_layer2.h_taus_gaus = torch.nn.Parameter( torch.randn( self.h_size2 ), requires_grad = rnn_settings['h_tau_trainable'][1] )
        w = torch.zeros( ( self.h_size2,self.h_size3 ), dtype = torch.float32 ) 
        ind = torch.where( torch.rand( self.h_size2,self.h_size3 ) < rnn_settings['p_rec'][1] )
        w[ ind[0], ind[1] ]  = torch.normal( torch.zeros( len(ind[0]) ), torch.ones( len(ind[0] ) ) ) * rnn_settings['W_ff_scalar'][1]
        if self.apply_dale[1] == True:
            w = torch.abs(w)
        self.h_layer2.wff_23 = torch.nn.Parameter( w ,requires_grad = rnn_settings['W_ff_trainable'][1] )
        self.h_layer2.mff_23 = torch.relu( self.h_layer2.mask.clone() )
    
        # set up feedback matrices/masks for layer 2 to 1 fb
        w = torch.zeros( ( self.h_size2,self.h_size1 ), dtype = torch.float32 ) 
        ind = torch.where( torch.rand( self.h_size2,self.h_size1 ) < rnn_settings['p_rec'][1] )
        w[ ind[0], ind[1] ]  = torch.normal( torch.zeros( len(ind[0]) ), torch.ones( len(ind[0] ) ) ) * rnn_settings['W_fb_scalar'][0] # 0 ind because only n-1 values in this list...
        if self.apply_dale[1] == True:
            w = torch.abs(w)
        self.h_layer2.wfb_21 = torch.nn.Parameter( w,requires_grad = rnn_settings['W_fb_trainable'][0] )        
        # mask for fb - so only exc units pass info
        self.h_layer2.mfb_21 = torch.relu( self.h_layer2.mask.detach().clone() ) 

        #--------------------------------
        # 3rd layer - no ff, just fb
        #--------------------------------
        self.h_tau3 = rnn_settings['h_tau'][2]   
        self.h_layer3 = HiddenLayer( rnn_settings,2 )
        self.h_layer3.h_taus_gaus = torch.nn.Parameter( torch.randn( self.h_size3 ), requires_grad = rnn_settings['h_tau_trainable'][2] )
        w = torch.zeros( ( self.h_size3,self.h_size2 ), dtype = torch.float32 ) 
        ind = torch.where( torch.rand( self.h_size3,self.h_size2 ) < rnn_settings['p_rec'][2] )
        w[ ind[0], ind[1] ]  = torch.normal( torch.zeros( len(ind[0]) ), torch.ones( len(ind[0] ) ) ) * rnn_settings['W_fb_scalar'][1] # 1 ind because only n-1 values in this list...
        if self.apply_dale[2] == True:
            w = torch.abs(w)
        self.h_layer3.wfb_32 = torch.nn.Parameter( w,requires_grad = rnn_settings['W_fb_trainable'][1] )        
        self.h_layer3.mfb_32 = torch.relu( self.h_layer3.mask.detach().clone() ) 

    #--------------------------------
    # define how stimulus inputs propagate through
    # the network...build a stack of the network states
    #--------------------------------    
    def forward(self, stims, cues):
        
        """
        Define how inputs propogate through the network by making a stack 
        of states...
            INPUT
                stims: [timepoints x batch_size(num trials) x inp_size]
                
            OUTPUT
                stacked_states: [seq_len x batch size x hidden_size], stack of hidden layer status
        """

        # initial states of hidden layers - batch size x h_size of small 
        # randn nums, then run through activation func for initial "firing rate"
        # the lists will store the state of each layer at each timestep
        h1 = []
        next_x1 = torch.randn( ( stims.shape[1], self.h_size1 ), dtype=torch.float32 )/100 
        r1 = self.act_func( next_x1 ) 
        
        h2 = []
        next_x2 = torch.randn( ( stims.shape[1], self.h_size2 ), dtype=torch.float32 )/100 
        r2 = self.act_func( next_x2 ) 
        
        h3 = []
        next_x3 = torch.randn( ( stims.shape[1], self.h_size3 ), dtype=torch.float32 )/100 
        r3 = self.act_func( next_x3 ) 
        
        # then a few lists to store ff and fb
        ff12 = []
        ff23 = []
        fb21 = []
        fb32 = []
        
        # scale taus and pass through act function...can do outside of the t
        # loop 
        h_taus_sig1 = torch.sigmoid( self.h_layer1.h_taus_gaus ) * ( self.h_tau1[1] - self.h_tau1[0] ) + self.h_tau1[0]
        h_taus_sig2 = torch.sigmoid( self.h_layer2.h_taus_gaus ) * ( self.h_tau2[1] - self.h_tau2[0] ) + self.h_tau2[0]
        h_taus_sig3 = torch.sigmoid( self.h_layer3.h_taus_gaus ) * ( self.h_tau3[1] - self.h_tau3[0] ) + self.h_tau3[0]
            
        # then loop over time start at 1st timepoint even though already initialized
        # values of x,r at time 0 above just so the ouput is of len T...can think of the 
        # initialization above as the state of the network at time -1, before the start
        # of a trial and rN isn't updated till the end of this loop 
        for t in range( 0,stims.shape[0] ):
            
            # pass stimulus at time t to input layer to get the input
            # to the first layer...
            stim_inp = self.inp_layer( stims[t,:,:] )

            # then get the feedback from layer 2 to layer 1 after applying the mask
            # to ensure that inh units in layer 2 don't contribute to the fb. This is done
            # by zeroing out the rows in the fb weight matrix before multiplying by 
            # the current activation levels in each r2 unit on each trial in the batch...
            # r2 is a batch_size x h_size2 matrix, and torch.matmul( self.h_layer2.mfb_21,self.h_layer2.wfb_21 )
            # is a h_size2 by h_size1 matrix and the matmul of mask and weights does the 
            # zeroing out of the rows corresponding the h_layer2 inh units
            if self.apply_dale[1] == True:
                fb_21 = torch.matmul( r2, torch.matmul( self.h_layer2.mfb_21,torch.relu( self.h_layer2.wfb_21 ) ) )
            else:
                fb_21 = torch.matmul( r2, torch.matmul( self.h_layer2.mfb_21,self.h_layer2.wfb_21 ) )
                
            # update currents for next time step using
            # stimulus input, current state of layer1 (and fb from layer2 and 
            # add scaled randn noise
            # Note that eval hidden layer using a call to matmul layer weights by last state
            # (self.h_layer1( r1 )) (i.e. state before 
            # the updating that is happening at the current time step). Mask for Dale's 
            # applied in the h_layer forward method...
            if (self.task_type == 'rdk_repro_cue' & self.cue_layer_num == 1):
                cue_input1 = self.cue_layer( cues[t,:,:] )
            else:
                cue_input1 = 0
            
            next_x1 = ( ( 1-( self.dt/h_taus_sig1 ) ) * next_x1 ) + ( ( self.dt/h_taus_sig1 ) * ( stim_inp + self.h_layer1( r1 ) + fb_21 + cue_input1 ) )
            next_x1 += torch.randn( self.h_size1, dtype=torch.float32 ) * self.noise

            # now move on to layer 2 - it will get ff input from layer 1 and 
            # feedback from layer 3
            
            # feedforward from layer 1
            if self.apply_dale[0] == True:
                ff_12 = torch.matmul( r1, torch.matmul( self.h_layer1.mff_12,torch.relu( self.h_layer1.wff_12 ) ) )
            else:
                ff_12 = torch.matmul( r1, torch.matmul( self.h_layer1.mff_12,self.h_layer1.wff_12 ) )
                
            # feedback from layer 3
            if self.apply_dale[2] == True:
                fb_32 = torch.matmul( r3, torch.matmul( self.h_layer3.mfb_32,torch.relu( self.h_layer3.wfb_32 ) ) )
            else:
                fb_32 = torch.matmul( r3, torch.matmul( self.h_layer3.mfb_32,self.h_layer3.wfb_32 ) )

            # next x for layer 2
            if (self.task_type == 'rdk_repro_cue' & self.cue_layer_num == 2):
                cue_input2 = self.cue_layer( cues[t,:,:] )
            else:
                cue_input2 = 0
            
            next_x2 = ( ( 1-( self.dt/h_taus_sig2 ) ) * next_x2 ) + ( ( self.dt/h_taus_sig2 ) * ( ff_12 + self.h_layer2( r2 ) + fb_32 + cue_input2 ) )
            next_x2 += torch.randn( self.h_size2, dtype=torch.float32 ) * self.noise                
                
            # now move on to layer 3 - it will get ff input from layer 2 and
            # no feedback...
            
            # feedforward from layer 2
            if self.apply_dale[1] == True:
                ff_23 = torch.matmul( r2, torch.matmul( self.h_layer2.mff_23,torch.relu( self.h_layer2.wff_23 ) ) )
            else:
                ff_23 = torch.matmul( r2, torch.matmul( self.h_layer2.mff_23,self.h_layer2.wff_23) )
                
            # next x for layer 3
            if (self.task_type == 'rdk_repro_cue' & self.cue_layer_num == 3):
                cue_input3 = self.cue_layer( cues[t,:,:] )
            else:
                cue_input3 = 0
                
            next_x3 = ( ( 1-( self.dt/h_taus_sig3 ) ) * next_x3 ) + ( ( self.dt/h_taus_sig3 ) * ( ff_23 + self.h_layer3( r3 ) + cue_input3 ) )
            next_x3 += torch.randn( self.h_size3, dtype=torch.float32 ) * self.noise       
            
            # after the full sweep is done, then compute next "firing rates" 
            # by passing next_xN through the activation function
            r1 = self.act_func( next_x1 )
            r2 = self.act_func( next_x2 )
            r3 = self.act_func( next_x3 )
            
            # then store the state of each layer in our list...
            h1.append( r1 )
            h2.append( r2 )
            h3.append( r3 )
            
            # store the ff and fb
            ff12.append( ff_12 )
            ff23.append( ff_23 )
            fb21.append( fb_21 )
            fb32.append( fb_32 )

        # stack states and return...
        return torch.stack( h1, dim=0 ), torch.stack( h2, dim=0 ), torch.stack( h3, dim=0 ),\
                torch.stack( ff12, dim=0 ), torch.stack( ff23, dim=0 ), torch.stack( fb21, dim=0 ),\
                torch.stack( fb32, dim=0 ), h_taus_sig1, h_taus_sig2, h_taus_sig3

#--------------------------------
# make the model object by calling the 
# recurrent object and adding an output
# layer. 
#--------------------------------
class rdkRNN(torch.nn.Module):
    
    #--------------------------------
    # init an instance with recurrent layer (input + hidden)
    # and output layer
    #--------------------------------
    def __init__(self, rnn_settings):
        super().__init__()

        # define recrrent layer (recurrent processing of input and hidden layer)
        self.recurrent_layer = RNN( rnn_settings )
        self.inp_size = rnn_settings['inp_size']            # number of inputs, last dim of a [time x batch size x num_inputs] matrix (where batch size == n_trials) 
        self.last_h_size = rnn_settings['h_size'][-1]       # size of last hidden layer
        self.n_h_layers = rnn_settings['n_h_layers']
        
        #define the output, or readout, layer. stock linear layer...
        # hard code these weight assignments for now as pretty standard...
        self.out_size = rnn_settings['out_size']
        self.W_out_trainable = rnn_settings['W_out_trainable']
        self.bias_out_trainable = rnn_settings['bias_out_trainable']
        self.W_out_scalar = rnn_settings['W_out_scalar']
        self.bias_out_scalar = rnn_settings['bias_out_scalar']
        W_out = torch.randn( ( self.out_size,self.last_h_size  ),dtype=torch.float32 ) * self.W_out_scalar
        bias_out = torch.randn( self.out_size,dtype=torch.float32 ) * self.bias_out_scalar
        
        self.output_layer = torch.nn.Linear( self.out_size,self.last_h_size  )
        self.output_layer.weight = torch.nn.Parameter( W_out,requires_grad=self.W_out_trainable )
        self.output_layer.bias = torch.nn.Parameter( bias_out,requires_grad=self.bias_out_trainable )
        
    #--------------------------------
    # setup the forward sweep of entire model
    # and compute output
    #--------------------------------
    def forward(self, inputs, cues):
        
        # get the state of the hidden layer units
        h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3 = self.recurrent_layer( inputs,cues )
        
        # run rand hidden state through the output layer
        output = self.output_layer( h3.float() )
        
        return output,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3
    
    #--------------------------------
    # Define loss function to use during model training 
    # This will determine the distance between model output and 
    # the desired (target) output during supervised learning
    # This is just the mean squared error (MSE), which you could
    # also implement with torch.nn.MSELoss()  
    #--------------------------------
    def mse_loss(self, outputs, targets):
        
        '''
        INPUT
            output: [time x trial_in_batch x output_size] model output of type torch.Tensor (float)
            target: [time x trial_in_batch] target of type torch.Tensor (float)
    
        OUTPUT
            loss, mean squared error (MSE) between outputs and targets. type torch.Tensor
        
        writing it out step by step for clarity, but more compact and harder to read is:
            torch.divide(torch.sum(torch.square(torch.subtract(torch.squeeze(output),target).flatten())),output.shape[0] * output.shape[1])
        '''
        # compute difference between output and target (squeeze in case output_size == 1)
        # flatten to vectorize the time x batch size (trials) matrices before passing to 
        # subsequent ops
        o_t_diff = torch.subtract( torch.squeeze(outputs),torch.squeeze(targets) ).flatten()
    
        # square the diff 
        o_t_sq_diff = torch.square( o_t_diff )
    
        # sum of squares
        o_t_ss_diff = torch.sum( o_t_sq_diff )
        
        # divide by number of data points to get mean squared error
        o_t_mse = torch.divide( o_t_ss_diff, o_t_diff.shape[0] )
        
        return o_t_mse
    
    #--------------------------------
    # Define loss function to use during model training 
    # Will weight based on trial count in each category...
    #--------------------------------
    def mse_weighted_loss(self, outputs, targets, s_label):
        
        '''
        INPUT
            output: [time x trial_in_batch x output_size] model output of type torch.Tensor (float)
            target: [time x trial_in_batch] target of type torch.Tensor (float)
    
        OUTPUT
            loss, mean squared error (MSE) between outputs and targets. type torch.Tensor

        '''
        
        # n_afc
        n_afc = outputs.shape[2]
        
        # timepoints * n_afc to scale mse...
        scale_mse = outputs.shape[0] * outputs.shape[2]
        
        # compute diff on a trial-by-trial basis. 
        o_t_diff = torch.subtract( torch.squeeze(outputs),torch.squeeze(targets) )
    
        # average error for each output channel - done to normalize by 
        # probability of occurence 
        o_t_mse = torch.zeros( n_afc )
        for stim in range( n_afc ):
            o_t_mse[ stim ] = torch.divide( torch.sum( torch.square( o_t_diff[:,s_label==stim,:] ) ), ( np.sum( s_label == stim ) * scale_mse ) )       

        # divide by number of data points to get mean squared error
        o_t_mse_out = torch.nanmean( o_t_mse ) 
        
        return o_t_mse_out
    
    
    
    