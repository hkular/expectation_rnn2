#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:29:11 2025

@author: hkular
"""

import subprocess
import itertools
import os


task_type = 'rdk_repro_cue'
classes = 'stim'
n_afc = 6
stim_amp_train = 1.0
stim_amp_eval = 1.0
stim_noise_train = 0.1
int_noise_train = 0.1
int_noise_eval = 0.1
stim_prob_eval = 1/n_afc
T = 210
num_cues = 2
cue_layer = 3
gpu = 1


# Varying args
time_or_xgen_vals = [0]
stim_noise_evals = [0.1, 0.6]
stim_prob_vals = [16,70]  
fb21_scalar_vals = [1.0, 0.7]
fb32_scalar_vals= [1.0, 0.7]
cue_on_vals = [0, 75]

# Fixed args for every run
base_cmd = [
    "python", "decode_multilayer.py",
    "--gpu", str(gpu),
    "--device", "gpu",
    "--classes", str(classes),
    "--task_type", str(task_type),
    "--stim_noise_train", str(stim_noise_train),
    "--stim_amp_train", str(stim_amp_train),
    "--stim_amp_eval", str(stim_amp_eval),
    "--int_noise_train", str(int_noise_train),
    "--int_noise_eval", str(int_noise_eval),
    "--cue_layer", str(cue_layer),

]
  

# Loop over all combinations
for time_or_xgen, cue_on, stim_prob, fb21_scalar, fb32_scalar, stim_noise_eval in itertools.product(
    time_or_xgen_vals, cue_on_vals, stim_prob_vals, fb21_scalar_vals, fb32_scalar_vals, stim_noise_evals
):
    # fn out for npz file to store decoding data
    if task_type == 'rdk_repro_cue':
        if time_or_xgen == 0:
            fn_out = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{stim_prob}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
        else:
            fn_out = f'decode_data/{task_type}_xgen_{classes}_{n_afc}nafc_stim_prob{stim_prob}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
    elif task_type == 'rdk_reproduction':
        if time_or_xgen == 0:
            fn_out = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{stim_prob}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
        else:
            fn_out = f'decode_data/{task_type}_xgen_{classes}_{n_afc}nafc_stim_prob{stim_prob}_stim_prob_eval-{int(stim_prob_eval * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
        
    if os.path.exists(fn_out):
        print(f"{fn_out} exists, skipping to next combo")
    else:
        if task_type == 'rdk_repro_cue':
            print(f"{task_type}, time_or_xgen={time_or_xgen}, cue_on={cue_on}, cue_layer={cue_layer}, stim_prob={stim_prob}, fb21_s={fb21_scalar}, fb32_s={fb32_scalar}")
        else:
            print(f"{task_type}, time_or_xgen={time_or_xgen}, stim_prob={stim_prob}, fb21_s={fb21_scalar}, fb32_s={fb32_scalar}")

        cmd = base_cmd + [
            "--time_or_xgen", str(time_or_xgen),
            "--cue_on", str(cue_on),
            "--stim_prob_train", str(stim_prob),
            "--stim_prob_eval", str(int(stim_prob_eval*100)),
            "--fb21_scalar", str(fb21_scalar),
            "--fb32_scalar", str(fb32_scalar),
            "--stim_noise_eval", str(stim_noise_eval)
        ]
    
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        

