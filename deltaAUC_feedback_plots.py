#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:41:02 2025

@author: hollykular
"""

import numpy as np
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
from scipy.stats import sem
#import scipy.stats as stats
from helper_funcs import *
from numpy import trapezoid
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
import pingouin as pg
#from statsmodels.stats.anova import AnovaRM
#import paramiko
#from io import BytesIO

#--------------------------
# Basic model params
#--------------------------
task_type = 'rdk_repro_cue'                         # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                                           # number of stimulus alternatives
T = 210                                             # timesteps in each trial
cue_on = 75                                          # 0(start) or 75(stim offset)
cue_layer = 1                                      # which layer gets the cue
stim_prob_train = 0.7
stim_prob_eval = stim_prob_train     
stim_amp_train = 1.0                                # can make this a list of amps and loop over... 
stim_amp_eval = 1.0
stim_noise_train = 0.1                              # magnitude of randn background noise in the stim channel
stim_noise_eval = 0.1
int_noise_train = 0.1                               # noise trained at 0.1
int_noise_eval = 0.1
weighted_loss = 0                                   # 0 = nw_mse l2 or 1 = weighted mse
num_cues = 2                                        # how many sr_scram
stim_on = 50                                        # timestep of stimulus onset
stim_dur = 25                                       # stim duration
cue_dur = T-cue_on                                  # on the rest of the trial
acc_amp_thresh = 0.8                                # to determine acc of model output: > acc_amp_thresh during target window is correct
h_size = 200                                        # how many units in a hidden layer
plots = False                                       # only plot if not run through terminal
n_layers =3
batch_size = 2000
out_size = n_afc

time_or_xgen = 0
w_size = 5
classes = 'stim'

settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_eval, 'stim_amp' : stim_amp_eval, 'stim_noise' : stim_noise_eval, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur}



# metrics
n_models = 20
n_stim_types = n_afc
decay_window = 50
sustain_window = 50
stim_offset = stim_on+stim_dur
decay_rates = np.zeros((n_models, n_layers, n_stim_types))
sustained_acc = np.zeros((n_models, n_layers, n_stim_types))

#--------------------------
# Which conditions to compare
#--------------------------
cue_onsets = [0, 75]
cue_layer = 3
stim_probs = [1/n_afc, 0.7]
fb21_scalars = [1.0,0.7,0.3,0.15,0]
fb32_scalars = [1.0,0.7,0.3,0.15,0]
valid_combos = [(1.0, 1.0)]  # always include both at 1.0
# fb21 varies, fb32=1.0
valid_combos += [(fb21, 1.0) for fb21 in fb21_scalars if fb21 != 1.0]
# fb32 varies, fb21=1.0
valid_combos += [(1.0, fb32) for fb32 in fb32_scalars if fb32 != 1.0]


results = []


# connect to remote server repository
#ssh = paramiko.SSHClient()
#ssh.load_system_host_keys()
#ssh.connect("128.59.20.250", username="holly", allow_agent=True,
#    look_for_keys=True)

#sftp = ssh.open_sftp()
#remote_base = '/home/holly/expectation_rnn2.0/'


for stim_prob in stim_probs:
    
    for cue_on in cue_onsets:
        
        
        for fb21_scalar, fb32_scalar in valid_combos:
    
             # load the correct model
             fn = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob * 100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_cueon-{cue_on}_ncues-{num_cues}_cuelayer-{cue_layer}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
             
            # remote_file = f"{remote_base}/{fn}"
             
             # Open remote file and read into memory
             #with sftp.file(remote_file, "rb") as f:
              #   buf = BytesIO(f.read())
             mod_data = np.load(fn, allow_pickle=True)
             
             # Process your data
             print(f"Loaded {fn}, keys: {mod_data.files}")
             
         
             # timing
             t = np.arange(0, T, T / mod_data['stim_acc'].shape[3])
             stim_offset_win = int(np.where(t == stim_offset)[0][0])
             decay_win = int(np.where(t==stim_offset+decay_window)[0][0])
             sustain_win = int(np.where(t==stim_offset+sustain_window)[0][0])
             
           
             for m in range(n_models):
                 for l in range(n_layers):
                         
                     if classes == 'cue':
                         # calculate AUC
                         area_one = trapezoid(mod_data['stim_acc'][m, l, 0, :][stim_offset_win:], t[stim_offset_win:])
                         area_two = trapezoid(mod_data['stim_acc'][m, l, 1, :][stim_offset_win:], t[stim_offset_win:])
                         
         
                         results.append({
                             'stim_prob': int(100*stim_prob),
                             'cue_on': cue_on,
                             'cue_layer': cue_layer,
                             'model': m,
                             'layer': l+1,
                             'AUC_one': area_one,
                             'AUC_two': area_two,
                             'delta_AUC': (area_one)-(area_two)
                             # 'decay': slope,
                             # 'peak': peak,
                             # 'sustain': sustained_acc[m,l,s]
                             
                             })
                     else:
                         # calculate AUC
                         area_exp = trapezoid(mod_data['stim_acc'][m, l, 0, :][stim_offset_win:], t[stim_offset_win:])
                         area_unexp = trapezoid(np.mean(mod_data['stim_acc'][m, l, 1:, :], axis = 0)[stim_offset_win:], t[stim_offset_win:])
                         
         
                         results.append({
                             'stim_prob': int(100*stim_prob),
                             'cue_on': cue_on,
                             'cue_layer': cue_layer,
                             'model': m,
                             'layer': l+1,
                             'fb21_scalar':fb21_scalar,
                             'fb32_scalar':fb32_scalar,
                             'AUC_exp': area_exp,
                             'AUC_unexp': area_unexp,
                             'delta_AUC': (area_exp)-(area_unexp)
                             
                             })


# Close connections
sftp.close()
ssh.close()

# create data frame
df = pd.DataFrame(results)

df['layer'] = df['layer'].astype(str)
df['stim_prob'] = df['stim_prob'].replace({16: 'Unbiased', 70: 'Biased'})
cueon_map = {0: 'Start', 75: 'Stim Offset'}
df['cue_on'] = df['cue_on'].map(cueon_map)
df['cue_on'] = pd.Categorical(
    df['cue_on'],
    categories=['Start', 'Stim Offset'],
    ordered=True
)
cueL_map = { 3: 'hLayer3'} # !! right now only doing cue layer 3
df['cue_layer'] = df['cue_layer'].map(cueL_map)
df['cue_layer'] = pd.Categorical(
    df['cue_layer'],
    categories=['hLayer1', 'hLayer3'],
    ordered=True
)




#--------------------------
# plot main effect of cueon when cue layer3
#--------------------------
df_ex = df[df['cue_layer']=='hLayer3']
# Set plot aesthetics
sns.set(style="ticks", context="talk")
# Initialize FacetGrid
g = sns.FacetGrid(df_ex, col="layer", col_wrap=3, sharey=True, height = 4, aspect = 1.2)
palette = sns.color_palette("viridis", 20)
custom_subset = [palette[i] for i in [16,10,4]]
# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="stim_prob",
    y="delta_AUC",
    hue="cue_on",
    palette = custom_subset,
    ci=None,
    errorbar=None,
    estimator=np.mean,
    dodge = True
)

# Add custom error bars
hue_order = ['Start', 'Stim Offset']
x_order = sorted(df_ex['stim_prob'].unique(), reverse=True)
n_hues = len(hue_order)
bar_width = 0.8
width_per_bar = bar_width / n_hues

for ax, layer in zip(g.axes.flat, sorted(df_ex['layer'].unique(), key=int)):
    subset = df_ex[df_ex['layer'] == layer]
    means = subset.groupby(['stim_prob', 'cue_on'])['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['stim_prob', 'cue_on'])['delta_AUC'].apply(sem).reset_index()

    for i, row in means.iterrows():
        prob = row['stim_prob']
        noise = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['stim_prob'] == prob) &
            (errors['cue_on'] == noise),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(prob)
        hloc = hue_order.index(noise)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
#g.set(ylim=(0, 3.5))
g.set_axis_labels("", "AUC Expected - Unexpected")
g.set_titles("Layer {col_name}")
g.add_legend(title='cue_on', bbox_to_anchor=(0.86, 0.5), loc='center left')

# Center shared x-axis label
plt.subplots_adjust(bottom=0.2, left=0.12)
#g.fig.text(0.5, 0.05, 'Stimulus Probability', ha='center', fontsize=14)
g.savefig(f"decode_data/plots/D_AUC_{classes}_stimprob_x_cueon_cuelayer3_feedback.png", format="png", bbox_inches="tight")
plt.show()



#--------------------------
# stats
#--------------------------

# make sure what's categorical is treated as such
df['cue_on'] = df['cue_on'].astype('category')
df['cue_layer'] = df['cue_layer'].astype('category')
df['layer'] = df['layer'].astype('category')
df['model'] = df['model'].astype('category')



cue3 = df[df['cue_layer']=="hLayer3"]

# cue_on x stim_prob x layer

mixed_cue3 = smf.mixedlm(
    "delta_AUC ~ C(cue_on) * C(stim_prob) * C(layer)",
    data=cue3,
    groups=cue3["model"],
    re_formula="~1"
).fit()
print(mixed_cue3.summary())

# cue_on x stim_prob
mixed_cue3 = smf.mixedlm(
    "delta_AUC ~ C(cue_on) * C(stim_prob)",
    data=cue3,
    groups=cue3["model"],
    re_formula="~1"
).fit()
print(mixed_cue3.summary())

# pairewise t-tests within each layer
cue3_biased = cue3[cue3['stim_prob']=='Biased']
for layer in cue3_biased['layer'].unique():
    tmp = cue3_biased[cue3_biased['layer'] == layer]

    ph = pg.pairwise_tests(
        dv="delta_AUC",
        within="cue_on",       # comparing Start vs Stim Offset
        subject="model",       # repeated measure = model
        data=tmp,
        padjust="bonf",        # adjust for multiple testing
        effsize="cohen",
        alternative = 'greater'
    )

    print(f"\nPost-hoc within {layer}:")
    print(ph)   # show full output so you see all columns


# at the end remind me which one we were working on  
print('\007') # make a sound   
print(f'finished {settings}')

fn_out = f"decode_data/plots/D_AUC_{classes}_stimprob_x_cueon_cuelayer3_feedback.npz"

np.savez( fn_out,df=df, mixed_cue3=mixed_cue3, ph=ph )

