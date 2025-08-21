#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:29:11 2025

@author: hkular
"""

import subprocess
import itertools


# Fixed args for every run
base_cmd = [
    "python", "decode_multilayer.py",
    "--gpu", "1",
    "--device", "gpu",
    "--classes", "stim",
    "--stim_noise_train", "0.1",
    "--stim_noise_eval", "0.1",
    "--stim_amp_train", "1.0",
    "--stim_amp_eval", "1.0",
    "--int_noise_train", "0.1",
    "--int_noise_eval", "0.1",
    "--cue_layer", "3",
    "--fb32_scalar", "1.0"

]

# Varying args
time_or_xgen_vals = [0, 1]
cue_on_vals = [0, 75]
#cue_layer_vals = [3]
cue_layer = 3
stim_prob_vals = [16, 70]  # stim_prob_eval will match
fb21_scalar_vals = [0.99,0.95,0.85]
#fb32_scalar_vals= [0.99,0.95,0.85]

# Loop over all combinations
for time_or_xgen, cue_on, stim_prob, fb21_scalar in itertools.product(
    time_or_xgen_vals, cue_on_vals, stim_prob_vals, fb21_scalar_vals
):
    
    
    print(f"time_or_xgen={time_or_xgen}, cue_on={cue_on}, cue_layer={cue_layer}, stim_prob={stim_prob}, fb21_s={fb21_scalar}")

    cmd = base_cmd + [
        "--time_or_xgen", str(time_or_xgen),
        "--cue_on", str(cue_on),
#        "--cue_layer", str(cue_layer),
        "--stim_prob_train", str(stim_prob),
        "--stim_prob_eval", str(stim_prob),
        "--fb21_scalar", str(fb21_scalar),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
