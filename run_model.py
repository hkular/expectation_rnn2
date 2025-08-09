#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 08:31:07 2025

@author: johnserences, jserences@ucsd.edu
@author2: hkular, hkular@ucsd.edu
"""

# imports
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
import argparse
# import class to make model, make hidden layer, and loss function
from expectation_rnn import *
# import class to make model inputs (defines different tasks, and can add other tasks easily using this framework)
from rdk_task import RDKtask



# example cmd line...
# python run_model.py --N 10 --gpu 1 --task rdk_repro_cue --n_afc 6 --int_noise 0.1 --ext_noise 0.1  --n_cues 2 --stim_amps 1.0 --stim_prob_mode biased --cue_onset 0 --cue_layer_num 3

# parse input args...
# parser = argparse.ArgumentParser(description='Training RDK Task RNNs')
# parser.add_argument('--N', required=False,type=int,
#         default='10', help="How many models?")
# parser.add_argument('--gpu', required=False,
#         default='0', help="Which gpu?")
# parser.add_argument('--task', required=True,
#         help="Which task: repro or repro_cue?")
# parser.add_argument('--n_afc', required=True,type=int, default='6',
#         help="How many stimulus alternatives?")
# parser.add_argument('--int_noise', required=True,type=float, default='0.1',
#         help="What additive (internal) noise do you want?")
# parser.add_argument('--ext_noise', required=True,type=float, default='0.1',
#         help="What stim (external) noise do you want?")
# parser.add_argument('--n_cues', required=True,type=int, default='2',
#         help="Number of s->r cues?")
# parser.add_argument('--stim_amps', nargs='+', type=float, default=[1.0],
#     help='List of stimulus amplitudes (e.g., --stim_amps 0.6 1.0)')
# parser.add_argument('--stim_prob_mode', type=str, default='biased',
#     help='Stimulus probability(e.g., biased or unbiased or both)')
# parser.add_argument('--cue_onset', required = True, type=int,
#     help='When does the cue come on (75 or 0)?')
# parser.add_argument('--cue_layer_num', required = True, type =int, default = '0',
#     help='Which layer receives cue (1,2,3)?')
# args = parser.parse_args()

# for easy debugging
parser = argparse.ArgumentParser(description='Training Sensory Recruit RNNs')
args = parser.parse_args()
args.N=1
args.gpu=0
args.task= 'rdk_repro_cue'
args.n_afc=6
args.int_noise=0.1
args.ext_noise=0.1
args.n_cues=2
args.stim_amps=[1.0]
args.stim_prob_mode='biased'
args.cue_onset=0
args.cue_layer_num=3


# check for available devices 
# (mps works for MX macs)
if torch.cuda.is_available():
    cuda_device = torch.device(f"cuda:{args.gpu}") 
    # set the device to default 
    torch.set_default_device(cuda_device)
    print(f'Using device: {torch.get_default_device()}')

elif torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    # set the device to default 
    torch.set_default_device(mps_device)
    print(f'Using device: {torch.get_default_device()}')

else:
    print("CUDA/MPS device not found. Using cpu")

#--------------------------------
# RNN Params
#-------------------------------

# explicitly set random seed manually for different model instantiations? 
set_rand_seed = 1

# task params
task_type = args.task             # task type (conceptually think of as a motion discrimination task...) 
n_afc = args.n_afc                # number of stimulus alternatives
T = 210                           # timesteps in each trial
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_amps = args.stim_amps        # list of amp of stim amps (will loop over these during training)
stim_noise = args.ext_noise       # magnitude of randn background noise in the stim channel
batch_size = 256                  # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct

                                  # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
if args.stim_prob_mode == 'biased':  
    stim_probs = [0.7]
elif args.stim_prob_mode == 'unbiased':
    stim_probs = [1/n_afc]
elif args.stim_prob_mode == 'both':
    stim_probs = [0.7, 1/n_afc]
            

# a few general params for all layers
weighted_loss = 0                 # 1 will weight loss for exp and unexp stims equally, 0 will put each trial on equal footing
noise = args.int_noise            # additive noise applied before activation function ( random normal * preact_n )
act_func = 'sigmoid'              # activation function linking x and r for each unit (either relu or sigmoid or tanh)
dt = 1                            # timestep

# input layer params
inp_size = n_afc                  # number of stim channels in Win
W_inp_scalar = 1.0                # scalar for the weights in the input matrix
bias_inp_scalar = 0.0             # 0 for no bias...
W_inp_trainable = False           # w_in trainable? 
bias_inp_trainable = False        # bias_in trainable?

# cue layer params - only used if task_type == 'rdk_repro_cue'
cue_layer_num = args.cue_layer_num
num_cues = args.n_cues            # number of s-r mappings
W_cue_scalar = 1.0                # scalar for the weights in the input matrix
bias_cue_scalar = 0.0             # 0 for no bias...
W_cue_trainable = False           # w_in trainable? 
bias_cue_trainable = False        # bias_in trainable?
if args.cue_onset == 75:
    cue_on = stim_on+stim_dur         # cue comes on when stim goes off
elif args.cue_onset == 0:
    cue_on = 0
else:
    print(f'Invalid cue_on,  are sure you want {cue_on}?')
cue_dur = T-cue_on                # cue stays on the rest of the trial

# general params for hidden layers - lists (or list of lists) 
# so that params can be set separately for each layer
# if desired...
n_h_layers = 3                            # number of layers
h_size = [ 200,200,200 ]                  # number of units in each hidden layer
h_tau = [ [4,25],[4,25],[4,25] ]          # range of taus - dt/tau determines decay time (how long prior state of a unit impacts current state)
h_tau_trainable = [ True,True,True ]      # train taus?
p_rec = [ 0.2,0.2,0.2 ]                   # probability of two hidden layer units forming a synapse
p_inh = [ 0.2,0.2,0.2 ]                   # probability that a hidden layer connection, if formed, will be inhibitory
apply_dale = [ True,True,True ]           # apply Dale's principle (i.e. exc and inh connections cannot change signs)
w_dist = 'gaus'                           # hidden layer weight distribution (gauss or gamma)
w_gain = [ 1.5,1.5,1.5 ]                  # gain on weights in hidden layer if w_dist == gauss
bias_scalar = [ 0.1,0.1,0.1 ]
W_h_trainable = [ True,True,True ]
bias_h_trainable = [ True,True,True ] 

# general params that define matrices 
# for feedforward projections
# ( n_layers - 1 ) entries because no feedback 
# from h_layer1 to w_in
W_ff_scalar = [ 1.0,1.0 ]                 # to scale FF weights on init
bias_ff_scalar = [ 0.1,0.1 ]              # to scale FF bias on init - if 0 and not trainable will be no bias...
W_ff_trainable = [ True,True ]               
bias_ff_trainable = [ True,True ]

# general params to regulate feedback...
W_fb_scalar = [ 1.0,1.0 ]             # scalar on randn initilization of weights/bias
bias_fb_scalar = [ 0.1,0.1 ]          # scale fb bias after initialization
W_fb_trainable = [ True,True ]        # are fb weights trainable 
bias_fb_trainable = [ True,True ]   # fb bias trainable

# output layer params
if ( task_type=='rdk_reproduction' ) | ( task_type=='rdk_repro_cue' ):
    out_size = n_afc                  # number of output channels
elif task_type=='rdk':
    out_size = 1
    
W_out_scalar = 0.01           # scale weights after initialization      
bias_out_scalar = 0           # scalar on out bias...set to zero and trainable == True if want to init this at 0
W_out_trainable = True        # allow output weights to be trained? 
bias_out_trainable = True     # allow bias to be trained? 

# model training params
loss_crit = 0.001             # stop training if loss < loss_crit 
acc_crit = 0.90               # or stop training if prediction acc is > acc_crit

#--------------------------------
# Model training params
#--------------------------------
iters = 200000               # number of training iterations (if criteria not met)
loss_update_step = 50        # output the current loss/acuracy every loss_update_step training iterations

# learning rate of optimizer function - step size during gradient descent
learning_rate = 0.01   

#--------------------------------
# loop over model instantiations - include
# an offset option if running this script on 
# multiple GPUs
#--------------------------------
n_models = args.N           
model_offset = 0

for m_num in range( model_offset,model_offset+n_models ):

    # for ensuring same weights/connectivity
    # for a given model instantiation at each 
    # desired distractor amp
    if set_rand_seed: 
        torch.manual_seed( m_num )
        np.random.seed( m_num )    
    
    # dict of params to init the network
    rnn_settings = {'task_type' : task_type, 'batch_size' : batch_size,'T' : T ,'stim_on' : stim_on, 'noise' : noise, 'dt' : dt, 'act_func' : act_func, 
                    'inp_size' : inp_size, 'W_inp_scalar' : W_inp_scalar, 'bias_inp_scalar' : bias_inp_scalar, 'W_inp_trainable' : W_inp_trainable,
                    'bias_inp_trainable' : bias_inp_trainable, 'cue_on': cue_on,'cue_layer_num': cue_layer_num,'num_cues' : num_cues, 'W_cue_scalar' : W_cue_scalar, 'bias_cue_scalar' : bias_cue_scalar,
                    'W_cue_trainable' : W_cue_trainable, 'bias_cue_trainable' : bias_cue_trainable,
                    'n_h_layers' : n_h_layers, 'h_size' : h_size, 'h_tau' : h_tau, 'h_tau_trainable' : h_tau_trainable,
                    'p_rec' : p_rec, 'p_inh' : p_inh, 'apply_dale' : apply_dale, 'w_dist' : w_dist, 'w_gain' : w_gain, 'bias_scalar' : bias_scalar, 'W_h_trainable' : W_h_trainable, 
                    'bias_h_trainable' : bias_h_trainable, 'W_ff_scalar' : W_ff_scalar, 'bias_ff_scalar' : bias_ff_scalar, 'W_ff_trainable' : W_ff_trainable, 
                    'bias_ff_trainable' : bias_ff_trainable, 'W_fb_scalar' : W_fb_scalar, 'bias_fb_scalar' : bias_fb_scalar,
                    'W_fb_trainable' : W_fb_trainable, 'bias_fb_trainable' : bias_fb_trainable, 'out_size' : out_size, 'W_out_trainable' : W_out_trainable,
                    'W_out_scalar' : W_out_scalar, 'bias_out_scalar' : bias_out_scalar, 'bias_out_trainable' : bias_out_trainable}


    #--------------------------------
    # loop over stim amplitudes
    #--------------------------------
    for s_amp in stim_amps :
        
        #--------------------------------
        # loop over stim probabilities
        #--------------------------------
        for s_prob in stim_probs: 
            
            # get the stim amp on this look for output file name - make an int 
            # just to keep file name cleaner...
            out_stim_amp = int( s_amp * 100 ) 
    
            # same for stim_prob
            out_stim_prob = int( s_prob * 100 )
    
            # same for stim noise
            out_stim_noise = int( stim_noise * 100 )
            
            # Init the network object - will have the same starting values each time...
            net = rdkRNN( rnn_settings )
        
            # init dict of task related params  - init here because it contains train_amp
            settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
                        'stim_prob' : s_prob, 'stim_amp' : s_amp, 'stim_noise' : stim_noise, 'batch_size' : batch_size, 
                        'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues': num_cues, 'cue_on' : cue_on, 'cue_dur' : cue_dur,}
            
            # create the task object
            task = RDKtask( settings )
            
                
            # Adam optimizer 
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            
            # for storing running average of loss and accuracy computed 
            # over the last batch of trials every loss_update_step trials
            running_loss = 0
            running_acc = 0 
            
            
            # output file name
            if task_type=='rdk_reproduction':
                if weighted_loss == 1:
                    fn = f'trained_models_{task_type}/repro_num_afc-{n_afc}_stim_prob-{out_stim_prob}_stim_amp-{out_stim_amp}_stim_noise-{out_stim_noise}_h_bias_trainable-{int(bias_h_trainable[0])}_modnum-{m_num}'
                else:
                    fn = f'trained_models_{task_type}/repro_num_afc-{n_afc}_stim_prob-{out_stim_prob}_stim_amp-{out_stim_amp}_stim_noise-{out_stim_noise}_h_bias_trainable-{int(bias_h_trainable[0])}_nw_mse_modnum-{m_num}'

            elif task_type=='rdk_repro_cue':
                if weighted_loss == 1:
                    fn = f'trained_models_{task_type}/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer_num}/reprocue_num_afc-{n_afc}_stim_prob-{out_stim_prob}_stim_amp-{out_stim_amp}_stim_noise-{out_stim_noise}_h_bias_trainable-{int(bias_h_trainable[0])}_modnum-{m_num}'
                else:
                    fn = f'trained_models_{task_type}/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer_num}/reprocue_num_afc-{n_afc}_stim_prob-{out_stim_prob}_stim_amp-{out_stim_amp}_stim_noise-{out_stim_noise}_h_bias_trainable-{int(bias_h_trainable[0])}_nw_mse_modnum-{m_num}'
                    
            # check to see if exists...if not, run, otherwise skip
            if os.path.exists(f'{fn}.pt') == False:       
            
    
                print(f'Training model {task_type}, mod_num {m_num}, n_afc {n_afc}, stim_prob {out_stim_prob}, stim_amp {out_stim_amp}, stim_noise {out_stim_noise}, cue_on {cue_on}, cue_layer {cue_layer_num}')
    
    
                # loop over number of model training iterations
                for i in range( iters ):
                    
                    # get a batch of inputs and targets
                    if task_type=='rdk':
                        inputs,s_label = task.generate_rdk_stim()  
                        targets = task.generate_rdk_target( s_label )
                        cues = np.zeros( (T, batch_size, n_stim_chans) )   #make dummy cues - not applied
    
                    elif task_type=='rdk_reproduction':
                        inputs,s_label = task.generate_rdk_reproduction_stim()  
                        targets = task.generate_rdk_reproduction_target( s_label )
                        cues = np.zeros( (T, batch_size, n_stim_chans) )   #make dummy cues - not applied
    
                    elif task_type=='rdk_repro_cue':
                        
                        # generate scrambled sr mapping for this model
                        if i == 0: 
                            sr_scram = task.gen_sr_scram()
                        
                        inputs,cues,s_label,c_label = task.generate_rdk_reproduction_cue_stim()  
                        targets = task.generate_rdk_reproduction_cue_target( s_label,sr_scram,c_label )
    
                    # zero out the gradient buffers before updating model params (e.g. Weights/biases)
                    optimizer.zero_grad()
                    
                    # pass inputs...get outputs and hidden layer states if 
                    # desired - valid cue if cue task desired, otherwise 0
                    outputs,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3 = net( inputs, cues ) 
    
                    # compute loss given current output and target 
                    # output on each trial in this batch           
                    if weighted_loss == 1: 
                        loss = net.mse_weighted_loss( outputs, targets, s_label )
                    else:
                        loss = net.mse_loss( outputs, targets )
                
                    # backprop the loss
                    loss.backward()
                
                    # single optimization step to update parameters
                    optimizer.step()        
                
                    # update running loss (just to keep track and to print out)
                    running_loss += loss.item()
                    
                    # Compute prediction accuracy (defined by the thresholds specified in settings dict)
                    if task_type=='rdk':
                        m_acc, tbt_acc = task.compute_acc( outputs,s_label )
                    
                    elif task_type=='rdk_reproduction':
                        m_acc, tbt_acc = task.compute_acc_reproduction( outputs,s_label )  
                    
                    elif task_type=='rdk_repro_cue':
                        m_acc, tbt_acc = task.compute_acc_reproduction_cue( outputs,s_label,targets )  
                        
                    running_acc += m_acc
                
                    # update about current loss and acc rate of model 
                    # every loss_update_step steps
                    if i % loss_update_step == loss_update_step-1:
                
                        # compute avg loss and avg acc over last loss_update_step iterations
                        running_loss /= loss_update_step
                        running_acc /= loss_update_step
                        
                        # print out to monitor training
                        print(f'Task {task_type}, mod_num {m_num}, num_afc {n_afc}, stim_prob {out_stim_prob}, stim_amp {out_stim_amp}, stim_noise {out_stim_noise}, cue_on {cue_on}, cue_layer {cue_layer_num}, Step {i+1}, Loss {running_loss:0.4f}, Acc {running_acc:0.4f}')
                
                        # see if we've reached criteria to stop training
                        if (running_loss < loss_crit) | (running_acc > acc_crit):
                            print('Training finished')
                            break
                        
                        # reset to zero before evaluating the loss and acc
                        # of the next loss_update_step iterations...
                        running_loss = 0
                        running_acc = 0            
                    
    
                # save out model...
                torch.save(net, f'{fn}.pt')
                
                # save out the sr mapping for this model if cue task
                if task_type == 'rdk_repro_cue':
                    tmp_scram = {}
                    for s in range( sr_scram.shape[0] ):
                        tmp_scram[s] = sr_scram[s].tolist()
                        
                    rnn_settings['sr_scram'] = tmp_scram
                
                # write params for this model to JSON file
                # after adding a field to include the number of training steps,
                # the current accuracy of model, and the current loss, etc
                rnn_settings['step'] = i+1     
                rnn_settings['running_acc'] = running_acc
                rnn_settings['running_loss'] = running_loss
                rnn_settings['acc_crit'] = acc_crit
                rnn_settings['weighted_loss'] = weighted_loss
                
                # dump params to json so that we have a complete record of all params
                # during model training                
                with open(f'{fn}.json', "w") as outfile:
                    json.dump((settings,rnn_settings), outfile)
            else:
                print(f'Exists, skipping {fn}')

        
