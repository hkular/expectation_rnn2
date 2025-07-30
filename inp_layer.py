#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:20:26 2025

@author: johnserences, jserences@ucsd.edu
"""

import torch
import numpy as np
from torch import pi
from scipy.special import i0, i1 
import matplotlib.pyplot as plt   # for debugging

#--------------------------------
# define input layer to map the stimulus
# into the first hidden layer
#--------------------------------
class InpLayer(torch.nn.Module):
    
    #--------------------------------
    # Grab params and init weights
    #--------------------------------
    def __init__(self, rnn_settings):
        
        """
        init params for the input layer...
        """
        
        # inherit
        super().__init__()
        
        # basic params defining the input layer
        self.inp_size = rnn_settings['inp_size']                # size of input layer 
        self.W_inp_scalar = rnn_settings['W_inp_scalar']        # scale weights and bias on init
        self.bias_inp_scalar = rnn_settings['bias_inp_scalar']
        self.h1_size = rnn_settings['h_size'][0]                # size of first hidden layer 
        
        # input layer weights and bias requires_grad flag (trainable)
        self.W_inp_trainable = rnn_settings['W_inp_trainable']
        self.weight = torch.nn.Parameter( torch.randn( (self.h1_size,self.inp_size ) ) * self.W_inp_scalar, requires_grad = self.W_inp_trainable ) 
           
        # bias stuff for input layer - if scalar is 0 and requires_grad==False then 
        # no bias term
        self.bias_inp_trainable = rnn_settings['bias_inp_trainable']
        self.bias = torch.nn.Parameter( torch.randn( self.h1_size ) * self.bias_inp_scalar,requires_grad = self.bias_inp_trainable )
    
    #--------------------------------
    # Define operations on forward sweep through
    # input layer
    #--------------------------------
    def forward(self, r_t):
        """
        take a tensor of input data and return
        a tensor of output data to map into the 
        first hidden layer...
        
        INPUT: r_t current 'firing rates' of hidden layer units
        
        OUTPUT: r_t * W_hid + bias
        """        
        
        # compute output for each trial in the current batch
        out = torch.matmul( r_t,self.weight.T ) + self.bias

        # return...
        return out
    