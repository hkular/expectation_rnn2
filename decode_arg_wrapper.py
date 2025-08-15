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
    "--device", "cpu",
    "--classes", "cue",
    "--stim_noise_train", "0.1",
    "--stim_noise_eval", "0.1",
    "--stim_amp_train", "1.0",
    "--stim_amp_eval", "1.0",
    "--int_noise_train", "0.1",
    "--int_noise_eval", "0.1"
]

# Varying args
time_or_xgen_vals = [0, 1]
cue_on_vals = [0, 75]
cue_layer_vals = [1, 3]
stim_prob_vals = [16, 70]  # stim_prob_eval will match

# Loop over all combinations
for time_or_xgen, cue_on, cue_layer, stim_prob in itertools.product(
    time_or_xgen_vals, cue_on_vals, cue_layer_vals, stim_prob_vals
):
    print(f"time_or_xgen={time_or_xgen}, cue_on={cue_on}, cue_layer={cue_layer}, stim_prob={stim_prob}")

    cmd = base_cmd + [
        "--time_or_xgen", str(time_or_xgen),
        "--cue_on", str(cue_on),
        "--cue_layer", str(cue_layer),
        "--stim_prob_train", str(stim_prob),
        "--stim_prob_eval", str(stim_prob)
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
