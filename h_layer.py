#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:57:36 2025

@author: johnserences, jserences@ucsd.edu
"""

import torch
import numpy as np
from torch import pi
import matplotlib.pyplot as plt   # for debugging

class HiddenLayer(torch.nn.Module):

    #--------------------------------
    # Create a hidden layer that supports sparse, random connections and 
    # that can enforce strict assignment of exc and inh units (Dale's principle)
    #--------------------------------   
    def __init__(self, rnn_settings,layer_num):
        
        # inherit
        super().__init__()
       
        # build dictionary of params for this layer
        self.h_size = rnn_settings['h_size'][layer_num]          # size of rand hidden layer
        self.p_rec = rnn_settings['p_rec'][layer_num]
        self.p_inh = rnn_settings['p_inh'][layer_num]
        self.apply_dale = rnn_settings['apply_dale'][layer_num]
        self.w_gain = rnn_settings['w_gain'][layer_num]
        self.W_h_trainable = rnn_settings['W_h_trainable'][layer_num]
        self.bias_h_trainable = rnn_settings['bias_h_trainable'][layer_num]
        self.bias_scalar = rnn_settings['bias_scalar'][layer_num]
        self.w_dist = rnn_settings['w_dist']
                
        # define cell types, build weight matrix, and build mask 
        # for each layer...
        self.exc, self.inh, self.exc_size, self.inh_size = self.define_cell_type()
                
        # Then assign weights/requires_grad flag (trainable)
        # to hidden weights/bias using p_rec,p_inh,Dale's principle, etc
        self.weight, self.mask = self.init_W_hid() 
        
        # bias stuff for hidden layer - if scalar 0 and requires_grad == False
        # then no bias...
        self.bias_h = torch.randn( self.h_size ) * self.bias_scalar
        self.bias = torch.nn.Parameter( self.bias_h,requires_grad = self.bias_h_trainable )
        
    #--------------------------------
    # Define cell types (exc/inh) and set up to either
    # follow Dale's principle or not...
    #--------------------------------
    def define_cell_type(self):

        """
        Randomly assign units as exc or inh based on desired
            proportion of inhibitory cells
            Do so in accordance with Dale's principle (or not)

        Returns
            exc: bool marking excitatory units
            inh: bool marking inhibitory units
            exc_size: number of excitatory units
            inh_size: number of inhibitory units

        """
        
        # If applying Dale's principle
        if self.apply_dale == True:
            
            # index of inh units based on desired proportion (p_inh)
            inh = torch.rand( self.h_size ) < self.p_inh
            
            # if not inhibitory, then excitatory
            exc = ~inh
            
            # number of inh units (inh_size) and 
            # number of exc units (exc_size)
            inh_size = len(torch.where( inh == True )[0])
            exc_size = self.h_size - inh_size

        # If not applying Dale's principle
        elif self.apply_dale == False:
            
            # no separate inhibitory units defined
            inh = torch.full( (self.h_size,),False ) 
            exc = torch.full( (self.h_size,),True )
            inh_size = 0
            exc_size = self.h_size

        return exc, inh, exc_size, inh_size

    
    #--------------------------------
    # Initialize custom weight matrix for hidden layer... 
    # apply Dale's principle if desired
    #--------------------------------
    def init_W_hid(self):

        '''
        Generate a connectivity weight matrix for the hidden layer W_hid
        using either a gaussian or gamma distribution.
        
        INPUTS:
            h_size: number of units in this hidden layer
            p_rec: probability of recurrent connections between units in the 
                hidden layer
            inh: [rand_size x rand_size] matrix indicating which connections should be 
                inhibitory
            w_dist: distribution to determine weights (gaussian or gamma)
            w_gain: scale factor for the weights
            apply_dale: apply Dale's principle? 
            note: can add more control over the gaussian/gamma distributions
                but here using values from Kim et al PNAS 2019
        
        OUTPUTS:
            w: [rand_size x rand_size] matrix of weights 
            m: mask of size [rand_size x rand_size] of 1's (excitatory units)
                  and -1's (for inhibitory units)
            bias: hidden layer bias
            
        Final weight matrix is w*mask as implemented in the recurrence method
        '''
        
        # Weight matrix [h_size x h_size] matrix
        w_h = torch.zeros( ( self.h_size, self.h_size ), dtype = torch.float32 )
        ind = torch.where( torch.rand( self.h_size, self.h_size ) < self.p_rec )
        
        if self.w_dist == 'gamma':
            w_h[ ind[0], ind[1] ] = np.random.gamma( 2, 0.003, len( ind[0] ) )

        elif self.w_dist == 'gaus':
            w_h[ ind[0], ind[1] ] = torch.normal( torch.zeros( len(ind[0]) ), torch.ones( len(ind[0] ) ) )
            w_h = w_h / torch.sqrt( torch.tensor( self.h_size ) * torch.tensor( self.p_rec ) ) * self.w_gain 
            
        # if using Dale's principle
        if self.apply_dale == True:
            
            # abs weights
            w_h = torch.abs( w_h )
        
            # mask matrix - set desired proportion of units to be inhibitory
            mask = torch.eye( self.h_size, dtype=torch.float32 )
            mask[ torch.where( self.inh==True )[0], torch.where( self.inh==True )[0] ] = -1
            
            # convert to torch param, not trainable 
            mask = torch.nn.Parameter( mask, requires_grad = False )
            
        # if not enforcing Dale's then just make an eye mask (no effect)
        # not going to be applying it anyway so really could be anything...
        else:
            
            mask = torch.eye( self.rand_size, dtype=torch.float32 )

        return torch.nn.Parameter( w_h, requires_grad = self.W_h_trainable ), mask       
    
    #--------------------------------
    # Define operations on forward sweep through
    # hidden layer
    #--------------------------------
    def forward(self, r_t):
        """
        take a tensor of input data and return
        a tensor of output data 
        
        INPUT: r_t current 'firing rates' of hidden layer units
        
        OUTPUT: r_t * W_hid + bias, either under Dale's principle or 
                    not...
        """        
        
        # make sure that exc/inh currents are maintained
        # if enforcing Dale's principle
        if self.apply_dale == True:
            
            # rectify weights: relu/max(0,weight) or abs (see Song et al., 2016 PLoS Comp Bio)             
            # then multiply by the mask  
            w = torch.matmul( torch.relu(self.weight), self.mask )
        
        else:
            
            # leave weight unchanged
            w = self.weight

        # compute output for each trial in the current batch
        out = torch.matmul( r_t,w.T ) + self.bias

        # return...
        return out
    