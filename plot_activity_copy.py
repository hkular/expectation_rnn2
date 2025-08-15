#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 07:25:59 2025

@author: johnserences
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# define train and eval amps
n_mods = 10
task_type = 'repro_dis_cue'
stim_type = 'focal'
n_stim_chans = [3,6,9,12]
stim_dis_weights = 'same'
dis_static_dynamic = 'dynamic'
num_cues = 2
cue_on = 0
h1_lam_l2 = 0.0
h2_lam_l2 = 0.0
sparse_type = 'weights'
fb_on = 0
ring_w_kappa_inh = 1.0
model_folder = 'trained_models'

dis_amps = [0,1]
# eval_amps = [0,1]

# ff and fb amps to eval
ff_amp = 1
fb_amps = [1]        

T = 240
r_size = 240
h_size = 480
stim_on = 20                      # timestep of stimulus onset (set above so both dicts)
stim_dur = 50                     # stim duration
t_step = 1
internal_noise_eval = 0.1
stim_amp = 1.0
stim_noise_train = 0.0
stim_noise_level_to_eval = 1.0

# how big of a window (in +- number of units from center point) to avg for mean responses
# ...note that a window of 3 will actually be a window of 7 (e.g. mid point +-3)
avg_win = 3

# compute delay interval
total_delay = int( ( T - stim_dur ) - ( stim_on + stim_dur ) )
n_t_steps = T // t_step

# to store eval acc
eval_acc = np.zeros( ( n_mods,len(dis_amps),len(dis_amps),len(fb_amps) ) ) 

# set up figures for data from the ring and rand layers (h1 and h2)
# fig, axs = plt.subplots( nrows=len(dis_amps), ncols=len(eval_amps), figsize=(7.5, 5) )
# fig2, axs2 = plt.subplots( nrows=len(dis_amps), ncols=len(eval_amps), figsize=(7.5, 5) )
fig, axs = plt.subplots( nrows=1, ncols=len(dis_amps), figsize=(7.5, 5) )
fig2, axs2 = plt.subplots( nrows=1, ncols=len(dis_amps), figsize=(7.5, 5) )

# colormap for plotting
cmap = plt.cm.viridis
cval = []
for line in range( len( n_stim_chans ) ):
    cval.append( cmap( line / ( len( n_stim_chans ) - 1) ) )

# loop over number of possible stims 
for n_afc_idx,n_afc in enumerate( n_stim_chans ):

    # loop over distractor amps
    for da_idx,da in enumerate( dis_amps ):    

        # loop over fb levels (in descending order)
        for fb_idx,fba in enumerate( fb_amps ):
        
            # for now just look at models that were evaled at the training dis amp
            ea = da
            ea_idx = da_idx 
            
            # file name to load
            fn = f'decode_data/{task_type}_{stim_type}_delay-{total_delay}_decode_nstims-{n_afc}_stim_noise-{stim_noise_train}_trnamp-{da}_evalamp-{ea}_distype-{stim_dis_weights}_ff-{ff_amp}_fb-{fba}_inh_kappa-{ring_w_kappa_inh}_cue_on-{cue_on}_h1_L2-{h1_lam_l2}_h2_L2-{h2_lam_l2}_ST-{sparse_type}_fb_on-{fb_on}_stim_noise_eval-{stim_noise_level_to_eval}_internal_noise-{internal_noise_eval}.npz'

            # load model and get relevant fields
            mod_data = np.load( fn,allow_pickle=True )
            if ( mod_data['n_tmpts'] != T):
                raise ValueError('Wrong number of timepoints')
                
            h1 = mod_data['avg_h1']
            h2 = mod_data['avg_h2']
            pd = mod_data['params_dict'][()]   #dict of params with inh/exc labels, taus, etc
            eval_acc[ :,da_idx,ea_idx,fb_idx ] = mod_data['eval_acc']
                
            # roll the h1 responses to common center based on stim - only do for 
            # focal stims because they have consistent structure
            if stim_type == 'focal':
                
                # loop over stim alternatives
                for sc in range( n_afc ):
                    
                    # should be same set of focal stims for all models, 
                    # so can just use model 0
                    peak = np.argmax( mod_data['inp_stims'][0,:,sc] )   # peak response
                    middle = h1.shape[-1] // 2                          # middle unit - use to center responses
                    
                    # roll the units so that the maximally response units to the current
                    # stim are in the middle...
                    h1[ :,sc,:,: ] = np.roll( h1[ :,sc,:,: ], middle-peak, axis=2 )
            
            # then take the mean across all stims after re-alignment
            h1 = np.mean( h1,axis=1 )
            
            # separate out inh and exc units from the ring layer
            h1_inh = h1[ :,:,pd[0]['inh1'] ]
            h1_exc = h1[ :,:,pd[0]['exc1'] ]
            
            # x-axis for plotting
            x = np.arange( 0,T,t_step )
            
            # plt mean response from exc units tuned to stim and from units
            # tuned ortho to the stim
            middle = h1_exc.shape[2] // 2
            md_stim = np.mean( np.mean( h1_exc[ :,:,middle-avg_win:middle+avg_win+1 ],axis=0 ),axis=1 )   # mean data in units tuned to stim
            semd_stim = sem( np.mean( h1_exc[ :,:,middle-avg_win:middle+avg_win+1 ],axis=2 ),axis=0 ) 
            md_ortho = np.mean( ( np.mean( h1_exc[ :,:,:avg_win+1 ],axis=2 ) + np.mean( h1_exc[ :,:,-avg_win: ],axis=2 ) ) / 2,axis=0 )   # mean data in units tuned to stim
            semd_ortho = sem( ( np.mean( h1_exc[ :,:,:avg_win+1 ],axis=2 ) + np.mean( h1_exc[ :,:,-avg_win: ],axis=2 ) ) / 2,axis=0 ) 
            axs[ da_idx ].plot(x, md_stim, c=cval[n_afc_idx], lw=2, label=n_afc)
            axs[ da_idx ].fill_between(x, md_stim - semd_stim, md_stim + semd_stim, color=cval[n_afc_idx], alpha=0.3)
            axs[ da_idx ].plot(x, md_ortho, lw=2, ls='--', c=cval[n_afc_idx], label=n_afc)
            axs[ da_idx ].fill_between(x, md_ortho - semd_ortho, md_ortho + semd_ortho, color=cval[n_afc_idx], alpha=0.3)

            # Add labels and legend
            axs[ da_idx ].set_ylim([-0.1,1.1])
            axs[ da_idx ].set_xlabel('Time Steps')
            axs[ da_idx ].set_ylabel('Response')
            axs[ da_idx ].set_title(f'DisAmp: {da}, EvalAmp: {ea}')

            # plt mean response from inh units tuned to stim and from units
            # tuned ortho to the stim
            middle = h1_inh.shape[2] // 2
            md_stim = np.mean( np.mean( h1_inh[ :,:,middle-avg_win:middle+avg_win+1 ],axis=0 ),axis=1 )   # mean data in units tuned to stim
            semd_stim = sem( np.mean( h1_inh[ :,:,middle-avg_win:middle+avg_win+1 ],axis=2 ),axis=0 ) 
            md_ortho = np.mean( ( np.mean( h1_inh[ :,:,:avg_win+1 ],axis=2 ) + np.mean( h1_inh[ :,:,-avg_win: ],axis=2 ) ) / 2,axis=0 )   # mean data in units tuned to stim
            semd_ortho = sem( ( np.mean( h1_inh[ :,:,:avg_win+1 ],axis=2 ) + np.mean( h1_inh[ :,:,-avg_win: ],axis=2 ) ) / 2,axis=0 ) 
            axs2[ da_idx ].plot(x, md_stim, lw=2, c=cval[n_afc_idx], label=n_afc)
            axs2[ da_idx ].fill_between(x, md_stim - semd_stim, md_stim + semd_stim, color=cval[n_afc_idx], alpha=0.3)
            axs2[ da_idx ].plot(x, md_ortho, lw=2, ls='--', c=cval[n_afc_idx], label=n_afc)
            axs2[ da_idx ].fill_between(x, md_ortho - semd_ortho, md_ortho + semd_ortho, color=cval[n_afc_idx], alpha=0.3)

            # Add labels and legend
            axs2[ da_idx ].set_ylim([-0.1,1.1])
            axs2[ da_idx ].set_xlabel('Time Steps')
            axs2[ da_idx ].set_ylabel('Response')
            axs2[ da_idx ].set_title(f'DisAmp: {da}, EvalAmp: {ea}')

plt.legend()
fig.tight_layout()
fig2.tight_layout()
plt.show()   

# Plot the model accuracy data 
# md is a [ dis_amp,eval_am,fb_amp ] matrix
md = np.mean( eval_acc,axis=0 )
semd = sem( eval_acc,axis=0 )

# one plot per training dis amp level  - start with models trained without 
# distractors
fig, axs = plt.subplots( nrows=1, ncols=len(dis_amps), figsize=(7.5, 5) )
d2plot = md[0,:,:]
sem2plot = semd[0,:,:]
for p in range( d2plot.shape[1] ):
    axs[ 0 ].plot(dis_amps,d2plot[:,p],'o-',lw=2,label=fb_amps[ p ])
    axs[ 0 ].fill_between(dis_amps, d2plot[:,p] - sem2plot[:,p], d2plot[:,p] + sem2plot[:,p], alpha=0.3)

# Add labels and legend
axs[ 0 ].set_ylim([0.0,1])
axs[ 0 ].set_xticks(dis_amps)
axs[ 0 ].set_xlabel('Evaluation Distractor Amplitude')
axs[ 0 ].set_ylabel('Accuracy')
axs[ 0 ].legend()
axs[ 0 ].set_title(f'Train With {int(dis_amps[0]*100)}% distractors')

# then plot models trained with 0.5 amp distractors
d2plot = md[1,:,:]
sem2plot = semd[1,:,:]
for p in range( d2plot.shape[1] ):
    axs[ 1 ].plot(dis_amps,d2plot[:,p],'o-',lw=2,label=fb_amps[ p ])
    axs[ 1 ].fill_between(dis_amps, d2plot[:,p] - sem2plot[:,p], d2plot[:,p] + sem2plot[:,p], alpha=0.3)

# Add labels and legend
axs[ 1 ].set_ylim([0.0,1])
axs[ 1 ].set_xticks(dis_amps)
axs[ 1 ].set_xlabel('Evaluation Distractor Amplitude')
axs[ 1 ].set_ylabel('Accuracy')
axs[ 1 ].legend()
axs[ 1 ].set_title(f'Train With {int(dis_amps[1]*100)}% distractors')

plt.tight_layout()
plt.show()   
