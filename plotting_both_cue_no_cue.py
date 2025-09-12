#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:46:30 2025

@author: hkular
"""

#--------------------------
# Imports
#--------------------------
import numpy as np
import matplotlib.pyplot as plt    
from scipy.stats import sem
import scipy.stats as stats
from helper_funcs import *
from numpy import trapezoid
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
import pingouin as pg
from itertools import product
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from rdk_task import RDKtask
import torch
from statannotations.Annotator import Annotator
from itertools import combinations


#--------------------------
# Data Gathering
#--------------------------

# load cue models
data = np.load('decode_data/plots/D_AUC_stim_repro_cue_sept.npz', allow_pickle = True)
results = data['results']
results_list = [item for item in results]  # Convert back to list
df_cue = pd.DataFrame(results_list)
#df = pd.DataFrame(data['results'])
df_cue['cue_layer'] = df_cue['cue_layer'].astype(str)
df_cue['layer'] = df_cue['layer'].astype(str)
df_cue['stim_prob'] = df_cue['stim_prob'].replace({16: 'Unbiased', 70: 'Biased'})
cueon_map = {0: 'Start', 75: 'Stim Offset'}
df_cue['cue_on'] = df_cue['cue_on'].map(cueon_map)
df_cue['cue_on'] = pd.Categorical(
    df_cue['cue_on'],
    categories=['Start', 'Stim Offset'],
    ordered=True
)


# load no cue models
data = np.load('decode_data/plots/D_AUC_stim_reproduction.npz', allow_pickle = True)
results = data['results']
results_list = [item for item in results]  # Convert back to list
df_nocue = pd.DataFrame(results_list)
df_nocue['layer'] = df_nocue['layer'].astype(str)
df_nocue['stim_prob'] = df_nocue['stim_prob'].replace({16: 'Unbiased', 70: 'Biased'})




# combine dataframes
# Add a source column
df_nocue['source'] = 'nocue'
df_cue['source'] = 'cue'
# All columns you want in the combined dataframe
all_cols = sorted(set(df_nocue.columns).union(df_cue.columns))

# Add missing columns with 0 to avoid dtype warning, remember to filter when plotting
for c in all_cols:
    if c not in df_nocue.columns:
        df_nocue[c] = 0 
    if c not in df_cue.columns:
        df_cue[c] = 0

# Reorder columns to be consistent
df_nocue = df_nocue[all_cols]
df_cue = df_cue[all_cols]

# Concatenate
df = pd.concat([df_nocue, df_cue], ignore_index=True) 

#--------------------------
# Fig 1A: task diagram
#--------------------------

x = np.arange(0, 100)
channels = np.zeros((6, len(x)))
channels[0, (x >= 25) & (x <= 40)] = 1  # step in channel 1
fig, axes = plt.subplots(6, 1, figsize=(6, 4), sharex=True)
for i, ax in enumerate(axes):
    ax.step(x, channels[i], where='post', color='blue', linewidth=2)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.text(-5, 0.6, f'{i+1}', ha='right', va='center', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False)
axes[0].text(32, 1.05, 'Sample', ha='center', va='bottom')
fig.text(0.06, 0.5, 'Input channel', va='center', rotation='vertical')
axes[-1].set_xlabel('Time steps')
#plt.savefig("iclr26_figs/Fig1A_task_diagram.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig1A_task_diagram.eps", format="eps", bbox_inches="tight")
plt.show()

#--------------------------
# Fig 1B: model architecture 
#--------------------------

# see Fig1B_original.ppt to make significant changes

#--------------------------
# Fig 1C: example input
#--------------------------


settings = {'task' : 'rdk_reproduction', 'n_afc' : 6, 'T' : 210, 'stim_on' : 50, 'stim_dur' : 25,
            'stim_prob' : 1/6, 'stim_amp' : 1.0, 'stim_noise' : 0.1, 'batch_size' : 40, 
            'acc_amp_thresh' : 0.8, 'out_size' : 6, 'num_cues':2, 'cue_on':0, 'cue_dur':210, 'rand_seed_bool':True, 'seed_num':42}

# create the task object
task = RDKtask( settings )
inputs,s_label = task.generate_rdk_reproduction_stim()

# plot inputs
plt.plot(inputs[:,37,:]) # t= 37 is first one which = orange
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.ylim(-0.5,1.3)
#plt.savefig("iclr26_figs/Fig1C_example_input.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig1C_example_input.eps", format="eps", bbox_inches="tight")
plt.show()

#--------------------------
# Fig 1D: example target
#--------------------------

settings = {'task' : 'rdk_reproduction', 'n_afc' : 6, 'T' : 210, 'stim_on' : 50, 'stim_dur' : 25,
            'stim_prob' : 1/6, 'stim_amp' : 1.0, 'stim_noise' : 0.1, 'batch_size' : 40, 
            'acc_amp_thresh' : 0.8, 'out_size' : 6, 'num_cues':2, 'cue_on':0, 'cue_dur':210, 'rand_seed_bool':True, 'seed_num':42}

# create the task object
task = RDKtask( settings )
inputs,s_label = task.generate_rdk_reproduction_stim()
targets = task.generate_rdk_reproduction_target( s_label )
# plot inputs
plt.plot(targets[:,37,:]) # t= 37 is first one which = orange
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.ylim(-0.5,1.3)
#plt.savefig("iclr26_figs/Fig1D_example_target.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig1D_example_target.eps", format="eps", bbox_inches="tight")
plt.show()

#--------------------------
# Fig 1E: example output
#--------------------------

settings = {'task' : 'rdk_reproduction', 'n_afc' : 6, 'T' : 210, 'stim_on' : 50, 'stim_dur' : 25,
            'stim_prob' : 1/6, 'stim_amp' : 1.0, 'stim_noise' : 0.1, 'batch_size' : 40, 
            'acc_amp_thresh' : 0.8, 'out_size' : 6, 'num_cues':2, 'cue_on':0, 'cue_dur':210, 'rand_seed_bool':True, 'seed_num':42}

# create the task object
task = RDKtask( settings )
inputs,s_label = task.generate_rdk_reproduction_stim()
targets = task.generate_rdk_reproduction_target( s_label )
cues = np.zeros( (task.T, task.batch_size, task.n_afc) )
# load example model
fn = 'trained_models_rdk_reproduction/timing_210/repro_num_afc-6_stim_prob-16_stim_amp-100_stim_noise-10_h_bias_trainable-1_nw_mse_modnum-0.pt'
model = load_model( fn,'cpu' )
with torch.no_grad():
    outputs,*_ = model( inputs,cues )

# plot inputs
plt.plot(outputs[:,37,:]) # t= 37 is first one which = orange
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.ylim(-0.5,1.3)
#plt.savefig("iclr26_figs/Fig1E_example_output.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig1E_example_output.eps", format="eps", bbox_inches="tight")
plt.show()



#--------------------------
# Fig 2A: AUC Metric
#--------------------------

# load example data
data = np.load('decode_data/plots/example_m_acc_rdk_reproduction_biased.npz')
over_acc = data['over_acc']
stim_acc = data['stim_acc']
m_over_acc = np.mean(over_acc,axis=0)
m_stim_acc = np.mean(stim_acc,axis=0)
sem_over_acc = sem(over_acc,axis=0)
sem_stim_acc = sem(stim_acc,axis=0)

# plot
sns.set(style="ticks", context="talk")
# Compute means
expected_mean = m_stim_acc[0, :]
unexpected_mean = np.mean(m_stim_acc[1:, :], axis=0)
expected_sem = sem_stim_acc[0, :]
unexpected_sem = np.mean(sem_stim_acc[1:, :], axis=0)
# Compute overlap
overlap = np.minimum(expected_mean, unexpected_mean)
# Colors
t = np.arange(0,210,5)
hex_c = ['#06D2AC', '206975']
expected_color = hex_c[0]
unexpected_color = hex_c[1] 
nonoverlap_color = "pink"
# Plot
plt.figure(figsize=(7,5))
plt.axvspan(50, 75, color="lightgray", alpha=0.4, zorder=0)
plt.fill_between(t, unexpected_mean, 0, color=unexpected_color, alpha=0.2, zorder=1)
plt.fill_between(t, expected_mean, overlap, where=expected_mean > unexpected_mean,
                 color=nonoverlap_color, alpha=0.4, zorder=2)
plt.fill_between(t, unexpected_mean, overlap, where=unexpected_mean > expected_mean,
                 color=nonoverlap_color, alpha=0.5, zorder=2)
plt.errorbar(t, m_stim_acc[0, :], sem_stim_acc[0, :],
    fmt=hex_c[0], label='Expected', capsize=3, lw=2)
plt.errorbar(t, np.mean(m_stim_acc[1:, :], axis=0),
    np.mean(sem_stim_acc[1:, :], axis=0),
    fmt=hex_c[1], label='Unexpected', capsize=3, lw=2)

plt.xlabel("Time step")
plt.ylabel("Decoding Accuracy")
sns.despine()
plt.legend(frameon=True, loc="lower right")
plt.tight_layout()
#plt.savefig("iclr26_figs/Fig2A_AUC.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig2A_AUC.eps", format="eps", bbox_inches="tight")
plt.show()


#--------------------------
# Fig 2B: RNN expectation with noisy stimuli
#--------------------------

df_ex = df[(df['fb21_scalar']==1.0) &
           (df['fb32_scalar']==1.0) & (df['source']=='nocue')]
# Compute delta AUC averaged across layers
agg = (
    df_ex.groupby(['stim_prob', 'stim_noise'])
    .agg(mean_auc=('delta_AUC', 'mean'),
         sem_auc=('delta_AUC', sem))
    .reset_index()
)
# aesthetics
sns.set(style="ticks", context="talk")
hue_order = list(np.unique(df['stim_noise']))
x_order = sorted(df_ex['stim_prob'].unique(), reverse=True)
discrete_palette = sns.color_palette('magma', n_colors=len(hue_order))
fig, ax = plt.subplots(figsize=(6, 4))

# Barplot
sns.barplot(
    data=agg,
    x="stim_prob", y="mean_auc",
    hue="stim_noise",
    palette=discrete_palette,
    order=x_order, hue_order=hue_order,
    errorbar=None, ax=ax
)
# Add custom error bars
bar_width = 0.8
n_hues = len(hue_order)
width_per_bar = bar_width / n_hues

for i, row in agg.iterrows():
    prob = row['stim_prob']
    noise = row['stim_noise']
    mean = row['mean_auc']
    err = row['sem_auc']
    xloc = x_order.index(prob)
    hloc = hue_order.index(noise)
    bar_center = xloc - bar_width/2 + width_per_bar/2 + hloc * width_per_bar
    ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# ---- 1. Pairwise comparisons of stim_noise within each stim_prob ----
pairs_noise = []
for prob in x_order:  # iterate each stim_prob level
    combs = list(itertools.combinations(hue_order, 2))
    for a, b in combs:
        pairs_noise.append(((prob, a), (prob, b)))

# ---- 2. Pairwise comparisons of stim_prob (collapsed across noise) ----
pairs_prob = []
combs = list(itertools.combinations(x_order, 2))
for a, b in combs:
    pairs_prob.append(((a, hue_order[0]), (b, hue_order[0])))  
    # use one hue category just to anchor the comparison

all_pairs = pairs_noise + pairs_prob

# ---- Annotator ----
annotator = Annotator(
    ax, all_pairs, data=df_ex,
    x="stim_prob", y="delta_AUC", hue="stim_noise",
    order=x_order, hue_order=hue_order
)

annotator.configure(test='t-test_paired', text_format='star',
                    loc='outside', comparisons_correction="holm", verbose=2)
annotator.apply_and_annotate()

# Labels and cleanup
ax.set_ylabel("Δ AUC")
ax.set_xlabel("")
sns.despine()
plt.ylim(-2,20)
# Legend (same style as before, no box)
ax.legend(title='Stimulus Noise',
          bbox_to_anchor=(0.9, 0.5),
          loc='center left',
          frameon=False)
plt.tight_layout()
#plt.savefig("iclr26_figs/Fig2B_D_AUC.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig2B_D_AUC.eps", format="eps", bbox_inches="tight")
plt.show()


# stats
aovrm = AnovaRM(df_ex, depvar='delta_AUC', subject='model', within=['stim_noise','stim_prob'], aggregate_func='mean')
res = aovrm.fit()
print(res)

#--------------------------
# Fig 3A: cue task diagram
#--------------------------

x = np.arange(0, 100)
channels = np.zeros(len(x))
channels[(x >= 25) & (x <= 40)] = 1  # sample step function

cue_timings = ['cue start', 'cue late']
cue_starts = [0, 41]  # shaded region start for each row

fig, axes = plt.subplots(2, 1, figsize=(8, 3), sharex=True)

for ax, cue_label, cue_start in zip(axes, cue_timings, cue_starts):
    # Plot sample step
    ax.step(x, channels, where='post', color='blue', linewidth=2)
    # Add shaded cue region
    ax.axvspan(cue_start, x[-1], color='orange', alpha=0.3)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_ylabel(cue_label, rotation=0, labelpad=40, va='center')
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False)

axes[-1].set_xlabel('Time steps')
plt.tight_layout()
#plt.savefig("iclr26_figs/Fig3A_task_diagram.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig3A_task_diagram.eps", format="eps", bbox_inches="tight")
plt.show()

#--------------------------
# Fig 3B: D_AUC cue onset x stim_noise x stim_prob x layer - V1
#--------------------------

df_ex = df[(df['fb32_scalar']==1.0) &
           (df['fb21_scalar']==1.0) & (df['source']=='cue')]
# Set plot aesthetics
sns.set(style="ticks", context="talk")
# Initialize FacetGrid
g = sns.FacetGrid(df_ex, row = "stim_noise", col="layer", sharey=True, height = 4, aspect = 1.2)
# Add custom error bars
hue_order = list(np.unique(df_ex['cue_on']))
x_order = list(sorted(df_ex['stim_prob'].unique(), reverse=True))
n_hues = len(np.unique(df_ex['cue_on']))
bar_width = 0.8
discrete_palette = sns.color_palette('viridis', n_colors=n_hues)

width_per_bar = bar_width / n_hues
# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="stim_prob",
    y="delta_AUC",
    hue="cue_on",
    palette = discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge = True,order=x_order,
    hue_order=hue_order
)
# sorted(df_ex['layer'].unique(), key=int)
for (noise, layer), ax in zip(
    [(n, l) for n in sorted(df_ex['stim_noise'].unique(), key=int)
            for l in sorted(df_ex['layer'].unique(), key=int)],
    g.axes.flat):
    subset = df_ex[(df_ex['stim_noise'] == noise) & (df_ex['layer'] == layer)]
    means = subset.groupby(['stim_prob', 'cue_on'], observed = False)['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['stim_prob', 'cue_on'], observed = False)['delta_AUC'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['stim_prob']
        h_ax = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['stim_prob'] == x_ax) &
            (errors['cue_on'] == h_ax),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
#g.set(ylim=(-4, 20))
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}, Noise {row_name}")
g.fig.text(0.07, 0.5, "AUC Expected - Unexpected", 
       va="center", rotation="vertical")
g.fig.text(0.5, 0.1, "Training", 
       va="center",)
g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')  
g.fig.text(0.5, 0.05, 'Main effects: Stimulus Noise (p<.001), Training (p<.001), Layer (p<.05),Interaction Cue Onset-Layer (p<.01)', ha='center', fontsize=14) 
# Center shared x-axis label
plt.subplots_adjust(bottom=0.2, left=0.12)
#plt.savefig("iclr26_figs/Fig3B_D_AUC_version1.svg", format="svg", bbox_inches="tight")
#plt.savefig("iclr26_figs/Fig3B_D_AUC_version1.eps", format="eps", bbox_inches="tight")
plt.show()


df_ex = df[(df['fb32_scalar']==1.0) &
           (df['fb21_scalar']==1.0) & (df['source']=='cue')]
df_ex = df_ex.copy()
categorical_cols = ['stim_noise', 'stim_prob', 'cue_on', 'layer']
for col in categorical_cols:
    df_ex[col] = df_ex[col].astype('category')

# stats
aovrm = AnovaRM(df_ex, depvar='delta_AUC', subject='model', within=['stim_noise','stim_prob', 'cue_on', 'layer'], aggregate_func='mean')
res = aovrm.fit()
print(res)

# pairwise comparisons


# stim_noise pairwise comparison collapsing all other factors
tukey = pairwise_tukeyhsd(
    endog=df_ex[(df['stim_noise']==0.6)]['delta_AUC'],        # dependent variable
    groups=df_ex['stim_noise'],       # factor to compare
    alpha=0.05
)
print(tukey.summary())


# which layer comparison is signficant - none
df_sub = df_ex#[(df_ex['cue_on']=='Start') & (df_ex['stim_prob']=='Biased') & (df_ex['stim_noise']==0.1)]
levels = df_sub['layer'].cat.categories
for a, b in combinations(levels, 2):
    g1 = df_sub[df_sub['layer']==a]['delta_AUC']
    g2 = df_sub[df_sub['layer']==b]['delta_AUC']
    stat, p = ttest_ind(g1, g2)
    print(f'Layer {a} vs {b}: t={stat:.3f}, p={p:.4f}')



#--------------------------
# Fig 3B: D_AUC cue onset x stim_noise x stim_prob x layer - V2
#--------------------------

df_ex = df[(df['fb32_scalar']==1.0) &
           (df['fb21_scalar']==1.0) & (df['source']=='cue') & (df['stim_noise']==0.6)]
# Set plot aesthetics
sns.set(style="ticks", context="talk")
# Initialize FacetGrid
g = sns.FacetGrid(df_ex, col="layer", sharey=True, height = 4, aspect = 1.2)
# Add custom error bars
hue_order = list(np.unique(df_ex['cue_on']))
x_order = list(sorted(df_ex['stim_prob'].unique(), reverse=True))
n_hues = len(np.unique(df_ex['cue_on']))
bar_width = 0.8
discrete_palette = sns.color_palette('viridis', n_colors=n_hues)

width_per_bar = bar_width / n_hues
# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="stim_prob",
    y="delta_AUC",
    hue="cue_on",
    palette = discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge = True,order=x_order,
    hue_order=hue_order
)
# sorted(df_ex['layer'].unique(), key=int)
for ( layer), ax in zip(
    [( l) for l in sorted(df_ex['layer'].unique(), key=int)],
    g.axes.flat):
    subset = df_ex[(df_ex['layer'] == layer)]
    means = subset.groupby(['stim_prob', 'cue_on'], observed = False)['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['stim_prob', 'cue_on'], observed = False)['delta_AUC'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['stim_prob']
        h_ax = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['stim_prob'] == x_ax) &
            (errors['cue_on'] == h_ax),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
#g.set(ylim=(-4, 20))
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}")
g.fig.text(0.07, 0.5, "AUC Expected - Unexpected", 
       va="center", rotation="vertical")
g.fig.text(0.5, 0.06, "Training", 
       va="center",)
g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')  
g.fig.text(0.1, 0.009, 'Main effects: Training (p<.001), Layer (p<.05), Interaction Cue Onset-Layer (p<.01)', ha='left', fontsize=14) 
# Center shared x-axis label
plt.subplots_adjust(bottom=0.2, left=0.12)
#plt.savefig(f"iclr26_figs/Fig3B_D_AUC_version2.svg", format="svg", bbox_inches="tight")
#plt.savefig(f"iclr26_figs/Fig3B_D_AUC_version2.eps", format="eps", bbox_inches="tight")
plt.show()


#--------------------------
# Fig 3B: D_AUC cue onset x stim_noise x stim_prob x layer - V3
#--------------------------

df_ex = df[(df['fb32_scalar']==1.0) &
           (df['fb21_scalar']==1.0) & (df['source']=='cue')]
agg = (
    df_ex.groupby(['stim_prob', 'stim_noise', 'layer', 'model'])
    .agg(mean_auc=('delta_AUC', 'mean'),
         sem_auc=('delta_AUC', sem))
    .reset_index()
)


# Set plot aesthetics
sns.set(style="ticks", context="talk")
# Initialize FacetGrid
g = sns.FacetGrid(agg, col="layer", sharey=True, height = 4, aspect = 1.2)
# Add custom error bars
hue_order = list(np.unique(agg['stim_noise']))
x_order = list(sorted(agg['stim_prob'].unique(), reverse=True))
n_hues = len(np.unique(agg['stim_noise']))
bar_width = 0.8
discrete_palette = sns.color_palette('magma', n_colors=n_hues)

width_per_bar = bar_width / n_hues
# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="stim_prob",
    y="mean_auc",
    hue="stim_noise",
    palette = discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge = True,order=x_order,
    hue_order=hue_order
)
# sorted(df_ex['layer'].unique(), key=int)
for ( layer), ax in zip(
    [( l) for l in sorted(agg['layer'].unique(), key=int)],
    g.axes.flat):
    subset = agg[ (agg['layer'] == layer)]
    means = subset.groupby(['stim_prob', 'stim_noise'], observed = False)['mean_auc'].mean().reset_index()
    errors = subset.groupby(['stim_prob', 'stim_noise'], observed = False)['mean_auc'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['stim_prob']
        h_ax = row['stim_noise']
        mean = row['mean_auc']
        err = errors.loc[
            (errors['stim_prob'] == x_ax) &
            (errors['stim_noise'] == h_ax),
            'mean_auc'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
#g.set(ylim=(-4, 20))
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}")
g.fig.text(0.07, 0.5, "Δ AUC", 
       va="center", rotation="vertical")
g.fig.text(0.5, 0.06, "Training", 
       va="center")
g.add_legend(title='Stimulus Noise', bbox_to_anchor=(0.86, 0.5), loc='center left')  
g.fig.text(0.1, 0.009, 'Main effects: Stimulus Noise (p<.001), Training (p<.001), Layer (p<.05)', ha='left', fontsize=14) 
# Center shared x-axis label
plt.subplots_adjust(bottom=0.2, left=0.12)
#plt.savefig(f"iclr26_figs/Fig3B_D_AUC_version3.svg", format="svg", bbox_inches="tight")
#plt.savefig(f"iclr26_figs/Fig3B_D_AUC_version3.eps", format="eps", bbox_inches="tight")
plt.show()

#--------------------------
# Fig 3B: D_AUC cue onset x stim_noise x stim_prob x layer - V4
#--------------------------

df_ex = df[(df['fb32_scalar']==1.0) &
           (df['fb21_scalar']==1.0) & (df['source']=='cue') & (df['stim_prob']=='Biased')]
# Set plot aesthetics
sns.set(style="ticks", context="talk")
# Initialize FacetGrid
g = sns.FacetGrid(df_ex,  col="layer", sharey=True, height = 4, aspect = 1.2)
# Add custom error bars
hue_order = list(np.unique(df_ex['cue_on']))
x_order = list(sorted(df_ex['stim_noise'].unique(), reverse=True))
n_hues = len(np.unique(df_ex['cue_on']))
bar_width = 0.8
discrete_palette = sns.color_palette('viridis', n_colors=n_hues)

width_per_bar = bar_width / n_hues
# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="stim_noise",
    y="delta_AUC",
    hue="cue_on",
    palette = discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge = True,order=x_order,
    hue_order=hue_order
)
# sorted(df_ex['layer'].unique(), key=int)
for (layer), ax in zip(
    [( l)for l in sorted(df_ex['layer'].unique(), key=int)],
    g.axes.flat):
    subset = df_ex[ (df_ex['layer'] == layer)]
    means = subset.groupby(['stim_noise', 'cue_on'], observed = False)['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['stim_noise', 'cue_on'], observed = False)['delta_AUC'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['stim_noise']
        h_ax = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['stim_noise'] == x_ax) &
            (errors['cue_on'] == h_ax),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
#g.set(ylim=(-4, 20))
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}")
g.fig.text(0.07, 0.5, "AUC Expected - Unexpected", 
       va="center", rotation="vertical")
g.fig.text(0.5, 0.1, "Stimulus Noise", 
       va="center",)
g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')  
g.fig.text(0.1, 0.05, 'Main effects: Stimulus Noise (p<.001),  Layer (p<.05), Interaction Cue Onset-Layer (p<.01)', ha='left', fontsize=14) 
# Center shared x-axis label
plt.subplots_adjust(bottom=0.2, left=0.12)
#plt.savefig(f"iclr26_figs/Fig3B_D_AUC_version4.svg", format="svg", bbox_inches="tight")
#plt.savefig(f"iclr26_figs/Fig3B_D_AUC_version4.eps", format="eps", bbox_inches="tight")
plt.show()


#--------------------------
# Fig 4A: D_AUC reduce feedback 2->1 V1
#--------------------------
df_ex = df[(df['fb32_scalar']==1.0) & (df['source']=='cue') & (df['stim_prob']=='Biased')]
# Average across stim_noise per model, layer, cue_on, fb21_scalar
df_ex = (df_ex.groupby(['model','layer','cue_on','fb21_scalar'], observed=False)['delta_AUC'].mean().reset_index())

# Set plot aesthetics
sns.set(style="ticks", context="talk")

# FacetGrid now only columns = layer
g = sns.FacetGrid(df_ex, col="layer", sharey=True, height=4, aspect=1.2)

# Barplot parameters
hue_order = list(np.unique(df_ex['cue_on']))
x_order = list(sorted(df_ex['fb21_scalar'].unique(), reverse=True))
n_hues = len(hue_order)
bar_width = 0.8
discrete_palette = sns.color_palette('viridis', n_colors=n_hues)
width_per_bar = bar_width / n_hues

# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="fb21_scalar",
    y="delta_AUC",
    hue="cue_on",
    palette=discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge=True,
    order=x_order,
    hue_order=hue_order
)

# Add error bars
for layer, ax in zip(sorted(df_ex['layer'].unique(), key=int), g.axes.flat):
    subset = df_ex[df_ex['layer'] == layer]
    means = subset.groupby(['fb21_scalar', 'cue_on'], observed=False)['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['fb21_scalar', 'cue_on'], observed=False)['delta_AUC'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['fb21_scalar']
        h_ax = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['fb21_scalar'] == x_ax) &
            (errors['cue_on'] == h_ax),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}")
g.fig.text(0.07, 0.5, "Δ AUC", va="center", rotation="vertical")
g.fig.text(0.5, 0.08, "Feedback Strength 2->1", va="center")
g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')
plt.subplots_adjust(bottom=0.2, left=0.12)
#plt.savefig(f"iclr26_figs/Fig4A_D_AUC_feedback21_version1.svg", format="svg", bbox_inches="tight")
#plt.savefig(f"iclr26_figs/Fig4A_D_AUC_feedback21_version1.eps", format="eps", bbox_inches="tight")
plt.show()

# stats

df_ex = df[(df['fb32_scalar']==1.0) & (df['source']=='cue')]
df_ex = df_ex.copy()
categorical_cols = [ 'cue_on', 'layer', 'fb21_scalar']
for col in categorical_cols:
    df_ex[col] = df_ex[col].astype('category')

# stats
aovrm = AnovaRM(df_ex, depvar='delta_AUC', subject='model', within=[ 'cue_on', 'layer', 'fb21_scalar'], aggregate_func='mean')
res = aovrm.fit()
print(res)

#--------------------------
# Fig 4B: D_AUC reduce feedback 3->2 V1
#--------------------------
df_ex = df[(df['fb21_scalar']==1.0) & (df['source']=='cue') & (df['stim_prob']=='Biased')]
# Average across stim_noise per model, layer, cue_on, fb21_scalar
df_ex = (df_ex.groupby(['model','layer','cue_on','fb32_scalar'], observed=False)['delta_AUC'].mean().reset_index())

# Set plot aesthetics
sns.set(style="ticks", context="talk")

# FacetGrid now only columns = layer
g = sns.FacetGrid(df_ex, col="layer", sharey=True, height=4, aspect=1.2)

# Barplot parameters
hue_order = list(np.unique(df_ex['cue_on']))
x_order = list(sorted(df_ex['fb32_scalar'].unique(), reverse=True))
n_hues = len(hue_order)
bar_width = 0.8
discrete_palette = sns.color_palette('viridis', n_colors=n_hues)
width_per_bar = bar_width / n_hues

# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="fb32_scalar",
    y="delta_AUC",
    hue="cue_on",
    palette=discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge=True,
    order=x_order,
    hue_order=hue_order
)

# Add error bars
for layer, ax in zip(sorted(df_ex['layer'].unique(), key=int), g.axes.flat):
    subset = df_ex[df_ex['layer'] == layer]
    means = subset.groupby(['fb32_scalar', 'cue_on'], observed=False)['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['fb32_scalar', 'cue_on'], observed=False)['delta_AUC'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['fb32_scalar']
        h_ax = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['fb32_scalar'] == x_ax) &
            (errors['cue_on'] == h_ax),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}")
g.fig.text(0.07, 0.5, "Δ AUC", va="center", rotation="vertical")
g.fig.text(0.5, 0.08, "Feedback Strength 3->2", va="center")
g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')
plt.subplots_adjust(bottom=0.2, left=0.12)

#plt.savefig(f"iclr26_figs/Fig4A_D_AUC_feedback32_version1.svg", format="svg", bbox_inches="tight")
#plt.savefig(f"iclr26_figs/Fig4A_D_AUC_feedback32_version1.eps", format="eps", bbox_inches="tight")
plt.show()

#--------------------------
# Fig 4B: D_AUC reduce feedback 3->2 V2
#--------------------------

df_ex = df[(df['fb21_scalar']==1.0) & (df['source']=='cue') &  (df['stim_prob']=='Biased')]
# Set plot aesthetics
sns.set(style="ticks", context="talk")
# Initialize FacetGrid
g = sns.FacetGrid(df_ex, row = "stim_noise", col="layer", sharey=True, height = 4, aspect = 1.2)
# Add custom error bars
hue_order = list(np.unique(df_ex['cue_on']))
x_order = list(sorted(df_ex['fb32_scalar'].unique(), reverse=True))
n_hues = len(np.unique(df_ex['cue_on']))
bar_width = 0.8
discrete_palette = sns.color_palette('viridis', n_colors=n_hues)

width_per_bar = bar_width / n_hues
# Map barplot onto each facet
g.map_dataframe(
    sns.barplot,
    x="fb32_scalar",
    y="delta_AUC",
    hue="cue_on",
    palette = discrete_palette,
    errorbar=None,
    estimator=np.mean,
    dodge = True,order=x_order,
    hue_order=hue_order
)
# sorted(df_ex['layer'].unique(), key=int)
for (noise, layer), ax in zip(
    [(n, l) for n in sorted(df_ex['stim_noise'].unique(), key=int)
            for l in sorted(df_ex['layer'].unique(), key=int)],
    g.axes.flat):
    subset = df_ex[(df_ex['stim_noise'] == noise) & (df_ex['layer'] == layer)]
    means = subset.groupby(['fb32_scalar', 'cue_on'], observed = False)['delta_AUC'].mean().reset_index()
    errors = subset.groupby(['fb32_scalar', 'cue_on'], observed = False)['delta_AUC'].apply(sem).reset_index()
    for i, row in means.iterrows():
        x_ax = row['fb32_scalar']
        h_ax = row['cue_on']
        mean = row['delta_AUC']
        err = errors.loc[
            (errors['fb32_scalar'] == x_ax) &
            (errors['cue_on'] == h_ax),
            'delta_AUC'
        ].values[0]
        xloc = x_order.index(x_ax)
        hloc = hue_order.index(h_ax)
        bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
        ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)

# Final plot cleanup
#g.set(ylim=(-4, 20))
g.set_axis_labels("", "")
g.set_titles("Layer {col_name}, Noise {row_name}")
g.fig.text(0.07, 0.5, "Δ AUC", 
       va="center", rotation="vertical")
g.fig.text(0.5, 0.1, "Feedback Strength 3->2", 
       va="center",)
g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')  
# Center shared x-axis label
plt.subplots_adjust(bottom=0.2, left=0.12)
#plt.savefig(f"iclr26_figs/Fig4B_D_AUC_feedback32_version2.svg", format="svg", bbox_inches="tight")
#plt.savefig(f"iclr26_figs/Fig4B_D_AUC_feedback32_version2.eps", format="eps", bbox_inches="tight")
plt.show()


df_ex = df[(df['fb21_scalar']==1.0) & (df['source']=='cue')]
df_ex = df_ex.copy()
categorical_cols = ['stim_noise', 'stim_prob', 'cue_on', 'layer']
for col in categorical_cols:
    df_ex[col] = df_ex[col].astype('category')

# stats
aovrm = AnovaRM(df_ex, depvar='delta_AUC', subject='model', within=['stim_noise', 'stim_prob', 'cue_on', 'layer', 'fb32_scalar'], aggregate_func='mean')
res = aovrm.fit()
print(res)









######### GRAVEYARD OF UNFINISHED CODE BELOW ###########


#--------------------------
# Plotting Eval Accuracy
#--------------------------


# ### MAIN EFFECT OF STIM NOISE ###
# df_ex = df[(df['fb32_scalar']==1.0) &
#            (df['fb21_scalar']==1.0) & (df['layer']=='1')]

# # Aggregate mean and SEM per stim_prob, stim_noise, and source
# stats_df = df_ex.groupby(['stim_prob', 'stim_noise', 'source'])['eval_acc'] \
#                  .agg(mean='mean', sem=sem).reset_index()

# ##PLOT
# sns.set(style="ticks", context="talk")
# hues = sorted(stats_df['stim_noise'].unique())
# palette = sns.color_palette("magma", n_colors=len(hues))
# palette = [palette[i] for i in range(len(palette)-1, -1, -1)]  # reverse
# g = sns.catplot(
#     data=stats_df,x='stim_prob',y='mean',hue='stim_noise',col='source',kind='bar',palette=palette,
#     errorbar=None,height=5,aspect=1)
# for ax, source_val in zip(g.axes.flat, sorted(stats_df['source'].unique())):
#     subset = stats_df[stats_df['source'] == source_val]
#     x_order = sorted(subset['stim_prob'].unique())
#     hue_order = sorted(subset['stim_noise'].unique())
#     n_hues = len(hue_order)
#     bar_width = 0.8
#     width_per_bar = bar_width / n_hues

#     for i, row in subset.iterrows():
#         xloc = x_order.index(row['stim_prob'])
#         hloc = hue_order.index(row['stim_noise'])
#         bar_center = xloc - bar_width/2 + width_per_bar/2 + hloc*width_per_bar
#         ax.errorbar(x=bar_center,y=row['mean'],yerr=row['sem'],fmt='none',c='black',capsize=5)
# g.set_axis_labels("Stimulus Probability", "Eval Accuracy")
# g.set_titles("{col_name} Task")
# g._legend.remove()
# plt.legend(title="Stimulus Noise", bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize = 14)
# plt.tight_layout()
# #plt.savefig(f"decode_data/plots/Eval_acc_{classes}_both_main.png", format="png", bbox_inches="tight")
# #plt.savefig(f"decode_data/plots/Eval_acc_{classes}_both_main.svg", format="svg", bbox_inches="tight")
# plt.show()

# ## STATS    
# # stats
# df_ex = df_ex.copy()
# df_ex['stim_noise'] = df_ex['stim_noise'].map({0.1: 'Low', 0.6: 'High'})

 
# aovrm = AnovaRM(df_ex, depvar='eval_acc', subject='model', within=['cue_on','stim_noise','stim_prob'])
# res = aovrm.fit()
# print(res)

# ### CUE MODELS - CUE EFFECT ###
# df_ex = df[(df['fb32_scalar']==1.0) &
#            (df['fb21_scalar']==1.0) & (df['layer']=='1') & (df['source']== 'cue')]

# # Aggregate mean and SEM per stim_prob, stim_noise, and source
# stats_df = df_ex.groupby(['stim_prob', 'stim_noise', 'cue_on'])['eval_acc'] \
#                  .agg(mean='mean', sem=sem).reset_index()

# ##PLOT
# sns.set(style="ticks", context="talk")
# hues = sorted(stats_df['stim_noise'].unique())
# palette = sns.color_palette("viridis", n_colors=len(hues))
# palette = [palette[i] for i in range(len(palette)-1, -1, -1)]  # reverse
# g = sns.catplot(
#     data=stats_df,x='stim_prob',y='mean',hue='stim_noise',col='cue_on',kind='bar',palette=palette,
#     errorbar=None,height=5,aspect=1)
# for ax, source_val in zip(g.axes.flat, sorted(stats_df['cue_on'].unique(), key = str)):
#     subset = stats_df[stats_df['cue_on'] == source_val]
#     x_order = sorted(subset['stim_prob'].unique())
#     hue_order = sorted(subset['stim_noise'].unique())
#     n_hues = len(hue_order)
#     bar_width = 0.8
#     width_per_bar = bar_width / n_hues

#     for i, row in subset.iterrows():
#         xloc = x_order.index(row['stim_prob'])
#         hloc = hue_order.index(row['stim_noise'])
#         bar_center = xloc - bar_width/2 + width_per_bar/2 + hloc*width_per_bar
#         ax.errorbar(x=bar_center,y=row['mean'],yerr=row['sem'],fmt='none',c='black',capsize=5)
# g.set_axis_labels("Stimulus Probability", "Eval Accuracy")
# g.set_titles("Cue Onset {col_name}")
# g._legend.remove()
# plt.legend(title="Stimulus Noise", bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize = 14)
# plt.tight_layout()
# #plt.savefig(f"decode_data/plots/Eval_acc_{classes}_cue_main.png", format="png", bbox_inches="tight")
# #plt.savefig(f"decode_data/plots/Eval_acc_{classes}_cue_main.svg", format="svg", bbox_inches="tight")
# plt.show()


### MAIN EFFECT OF FB SCALARS ###