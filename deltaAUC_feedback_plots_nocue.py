#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:41:02 2025

@author: hollykular
"""

import numpy as np
import matplotlib.pyplot as plt    # note: importing this in all files just for debugging stuff
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

#--------------------------
# Basic model params
#--------------------------
task_type = 'rdk_reproduction'                         # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                                           # number of stimulus alternatives
T = 210                                             # timesteps in each trial
stim_prob_train = 0.7
stim_prob_eval = 1/n_afc     
stim_amp_train = 1.0                                # can make this a list of amps and loop over... 
stim_amp_eval = 1.0
stim_noise_train = 0.1                              # magnitude of randn background noise in the stim channel
stim_noise_eval = 0.1
int_noise_train = 0.1                               # noise trained at 0.1
int_noise_eval = 0.1
weighted_loss = 0                                   # 0 = nw_mse l2 or 1 = weighted mse
stim_on = 50                                        # timestep of stimulus onset
stim_dur = 25                                       # stim duration
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
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size}



# metrics
n_models = 20
n_stim_types = n_afc
stim_offset = stim_on+stim_dur
#--------------------------
# Which conditions to compare
#--------------------------
stim_probs = [1/n_afc, 0.7]
fb21_scalars = [1.0,0.7]
fb32_scalars = [1.0,0.7]
stim_noises = [0.1,0.6]
valid_combos = list(product(fb21_scalars, fb32_scalars))

results = []

plots = False

# connect to remote server repository
#ssh = paramiko.SSHClient()
#ssh.load_system_host_keys()
#ssh.connect("128.59.20.250", username="holly", allow_agent=True,
#    look_for_keys=True)

#sftp = ssh.open_sftp()
#remote_base = '/home/holly/expectation_rnn2.0/'


for stim_prob in stim_probs:
    
        
    for stim_noise_eval in stim_noises:
    
        for fb21_scalar, fb32_scalar in valid_combos:
    
             # load the correct model
             if stim_prob == 0.7:
                 fn = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob * 100)}_stim_prob_eval-{int(stim_prob_eval*100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
             else:
                 fn = f'decode_data/{task_type}_decode_{classes}_{n_afc}afc_stim_prob{int(stim_prob * 100)}_stim_prob_eval-{int(stim_prob_eval*100)}_trnamp-{stim_amp_train}_evalamp-{stim_amp_eval}_trnnoise-{stim_noise_train}_evalnoise-{stim_noise_eval}_trnint-{int_noise_train}_evalint-{int_noise_eval}_T-{T}_nw_mse_fb21_s{fb21_scalar}_fb32_s{fb32_scalar}.npz'
             
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
           
             for m in range(n_models):
                 for l in range(n_layers):
                         
                     
                        # calculate AUC
                        area_exp = trapezoid(mod_data['stim_acc'][m, l, 0, :][stim_offset_win:], t[stim_offset_win:])
                        area_unexp = trapezoid(np.mean(mod_data['stim_acc'][m, l, 1:, :], axis = 0)[stim_offset_win:], t[stim_offset_win:])
                        
                        results.append({
                            'stim_prob': int(100*stim_prob),
                            'model': m,
                            'layer': l+1,
                            'fb21_scalar':fb21_scalar,
                            'fb32_scalar':fb32_scalar,
                            'AUC_exp': area_exp,
                            'AUC_unexp': area_unexp,
                            'delta_AUC': (area_exp)-(area_unexp),
                            'eval_acc': mod_data['m_acc'][m],
                            'stim_noise': stim_noise_eval
                            })

# Close connections
#sftp.close()
#ssh.close()

fn_out = f"decode_data/plots/D_AUC_{classes}_reproduction.npz"

np.savez( fn_out,results=results)


if plots:
    
    
    # load npz saved on fishee transferred to nc6
    data = np.load(f'decode_data/plots/D_AUC_{classes}_reproduction.npz', allow_pickle = True)
    results = data['results']
    results_list = [item for item in results]  # Convert back to list
    df = pd.DataFrame(results_list)
    df['layer'] = df['layer'].astype(str)
    df['stim_prob'] = df['stim_prob'].replace({16: 'Unbiased', 70: 'Biased'})

    
    #--------------------------
    # plot main effect of cue timing - eval acc
    #--------------------------
    df_ex = df[(df['fb32_scalar']==1.0) &
               (df['fb21_scalar']==1.0) & (df['layer']=='1')]
    # Set plot aesthetics
    sns.set(style="ticks", context="talk")
    # Initialize FacetGrid
    g = sns.FacetGrid(df_ex, col="stim_noise", col_wrap=2, sharey=True, height = 4, aspect = 1)
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
        y="eval_acc",
        hue="cue_on",
        palette = discrete_palette,
        errorbar=None,
        estimator=np.mean,
        dodge = True,order=x_order,
        hue_order=hue_order
    )
    for ax, stim_noise in zip(g.axes.flat, sorted(df_ex['stim_noise'].unique(), key=int)):
        subset = df_ex[df_ex['stim_noise'] == stim_noise]
        means = subset.groupby(['stim_prob', 'cue_on'], observed = False)['eval_acc'].mean().reset_index()
        errors = subset.groupby(['stim_prob', 'cue_on'], observed = False)['eval_acc'].apply(sem).reset_index()
    
        for i, row in means.iterrows():
            x_ax = row['stim_prob']
            h_ax = row['cue_on']
            mean = row['eval_acc']
            err = errors.loc[
                (errors['stim_prob'] == x_ax) &
                (errors['cue_on'] == h_ax),
                'eval_acc'
            ].values[0]
            xloc = x_order.index(x_ax)
            hloc = hue_order.index(h_ax)
            bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
            ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)
    
    # Final plot cleanup
    #g.set(ylim=(0, 3.5))
    g.set_axis_labels("", "Eval Accuracy")
    g.set_titles("Stimulus Noise {col_name}")
    g.add_legend(title='Cue Onset', bbox_to_anchor=(0.86, 0.5), loc='center left')
    
    # Center shared x-axis label
    plt.subplots_adjust(bottom=0.2, left=0.12)
    #g.fig.text(0.5, 0.05, 'Evaluation Accuracy', ha='center', fontsize=14)
    g.savefig(f"decode_data/plots/Eval_acc_{classes}_stimprob_x_cueon_cuelayer3_main.png", format="png", bbox_inches="tight")
    g.savefig(f"decode_data/plots/Eval_acc_{classes}_stimprob_x_cueon_cuelayer3_main.svg", format="svg", bbox_inches="tight")
    plt.show()
    
    # stats
    df_ex = df_ex.copy()
    df_ex['stim_noise'] = df_ex['stim_noise'].map({0.1: 'Low', 0.6: 'High'})

    # mixed = smf.mixedlm(
    #     "eval_acc ~ C(cue_on) + C(stim_noise) + C(stim_prob)",
    #     data=df_ex,
    #     groups=df_ex["model"],
    #     re_formula="~1"
    # ).fit()
    # print(mixed.summary())
    
   
    aovrm = AnovaRM(df_ex, depvar='eval_acc', subject='model', within=['cue_on','stim_noise','stim_prob'])
    res = aovrm.fit()
    print(res)
    
    # no post hoc comparison
    
   
    
    #--------------------------
    # plot eval acc - fb21_s = 1.0 ..reducing fb32 in biased models
    #--------------------------
    df_ex = df[(df['stim_prob']=='Biased') &
               (df['fb21_scalar']==1.0) & (df['layer']=='1')]
    # Set plot aesthetics
    sns.set(style="ticks", context="talk")
    # Initialize FacetGrid
    g = sns.FacetGrid(df_ex, col="stim_noise", col_wrap=2, sharey=True, height = 4, aspect = 1)
    # Add custom error bars
    hue_order = list(np.unique(df_ex['fb32_scalar']))
    x_order = list(sorted(df_ex['cue_on'].unique(), reverse=True))
    n_hues = len(np.unique(df_ex['fb32_scalar']))
    bar_width = 0.8
    discrete_palette = sns.color_palette('viridis', n_colors=n_hues)

    width_per_bar = bar_width / n_hues
    # Map barplot onto each facet
    g.map_dataframe(
        sns.barplot,
        x="cue_on",
        y="eval_acc",
        hue="fb32_scalar",
        palette = discrete_palette,
        errorbar=None,
        estimator=np.mean,
        dodge = True,order=x_order,
        hue_order=hue_order
    )
       
    for ax, stim_noise in zip(g.axes.flat, sorted(df_ex['stim_noise'].unique(), key=int)):
        subset = df_ex[df_ex['stim_noise'] == stim_noise]
        means = subset.groupby(['cue_on', 'fb32_scalar'], observed=False)['eval_acc'].mean().reset_index()
        errors = subset.groupby(['cue_on', 'fb32_scalar'], observed=False)['eval_acc'].apply(sem).reset_index()
    
        for i, row in means.iterrows():
            prob = row['cue_on']
            noise = row['fb32_scalar']
            mean = row['eval_acc']
            err = errors.loc[
                (errors['cue_on'] == prob) &
                (errors['fb32_scalar'] == noise),
                'eval_acc'
            ].values[0]
            xloc = x_order.index(prob)
            hloc = hue_order.index(noise)
            bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
            ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)
    
    # Final plot cleanup
    #g.set(ylim=(0, 3.5))
    g.set_axis_labels("", "Eval Accuracy")
    g.set_titles("Stimulus Noise {col_name}")
    g.add_legend(title='fb32_scalar', bbox_to_anchor=(0.9, 0.5), loc='center left')
    
    # Center shared x-axis label
    plt.subplots_adjust(bottom=0.2, left=0.12)
    #g.fig.text(0.5, 0.05, f'Eval Acc', ha='center', fontsize=14)
    #g.savefig(f"decode_data/plots/Eval_acc_{classes}_stimprob_x_cueon_cuelayer3_feedback32.png", format="png", bbox_inches="tight")
    plt.show()
    
    # stats
    df_ex.loc[:,"fb32_scalar"] = pd.Categorical(df_ex["fb32_scalar"], 
                                          categories=[1.0, 0.7], 
                                          ordered=True)
    
    aovrm = AnovaRM(df_ex, depvar='eval_acc', subject='model', within=['cue_on','fb32_scalar', 'stim_noise'])
    res = aovrm.fit()
    print(res)
    
  
    #--------------------------
    # plot AUC  - fb21_s = 1.0 ..reducing fb32 in biased models
    #--------------------------
    for noises in df['stim_noise'].unique():
        df_ex = df[(df['stim_prob']=='Biased') &
                   (df['fb21_scalar']==1.0) & (df['stim_noise']==noises)]
        # Set plot aesthetics
        sns.set(style="ticks", context="talk")
        # Initialize FacetGrid
        g = sns.FacetGrid(df_ex, col="layer", col_wrap=3, sharey=True, height = 4, aspect = 1.2)
        # Add custom error bars
        hue_order = list(np.unique(df_ex['fb32_scalar']))
        x_order = list(sorted(df_ex['cue_on'].unique(), reverse=True))
        n_hues = len(np.unique(df_ex['fb32_scalar']))
        bar_width = 0.8
        discrete_palette = sns.color_palette('viridis', n_colors=n_hues)
    
        width_per_bar = bar_width / n_hues
        # Map barplot onto each facet
        g.map_dataframe(
            sns.barplot,
            x="cue_on",
            y="delta_AUC",
            hue="fb32_scalar",
            palette = discrete_palette,
            errorbar=None,
            estimator=np.mean,
            dodge = True,order=x_order,
            hue_order=hue_order
        )
           
        for ax, layer in zip(g.axes.flat, sorted(df_ex['layer'].unique(), key=int)):
            subset = df_ex[df_ex['layer'] == layer]
            means = subset.groupby(['cue_on', 'fb32_scalar'], observed = True)['delta_AUC'].mean().reset_index()
            errors = subset.groupby(['cue_on', 'fb32_scalar'], observed = True)['delta_AUC'].apply(sem).reset_index()
        
            for i, row in means.iterrows():
                prob = row['cue_on']
                noise = row['fb32_scalar']
                mean = row['delta_AUC']
                err = errors.loc[
                    (errors['cue_on'] == prob) &
                    (errors['fb32_scalar'] == noise),
                    'delta_AUC'
                ].values[0]
                xloc = x_order.index(prob)
                hloc = hue_order.index(noise)
                bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
                ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)
        
        # Final plot cleanup
        g.set(ylim=(0, 40))
        g.set_axis_labels("", "AUC Expected - Unexpected")
        g.set_titles("Layer {col_name}")
        g.add_legend(title='Feedback 3->2', bbox_to_anchor=(0.89, 0.5), loc='center left')
        
        # Center shared x-axis label
        plt.subplots_adjust(bottom=0.2, left=0.12)
        g.fig.text(0.5, 0.05, f'Stimulus Noise {noises}', ha='center', fontsize=14)
        g.savefig(f"decode_data/plots/D_AUC_{classes}_cueon_cuelayer3_feedback32_stimnoise{int(noises*100)}.png", format="png", bbox_inches="tight")
        g.savefig(f"decode_data/plots/D_AUC_{classes}_cueon_cuelayer3_feedback32_stimnoise{int(noises*100)}.svg", format="svg", bbox_inches="tight")
        plt.show()
        
        # stats
        # df_ex.loc[:,"fb32_scalar"] = pd.Categorical(df_ex["fb32_scalar"], 
        #                                       categories=[1.0, 0.7, 0.3], 
        #                                       ordered=True)
        # mixed = smf.mixedlm(
        #     "delta_AUC ~ C(cue_on) + C(layer) + C(fb32_scalar)",
        #     data=df_ex,
        #     groups=df_ex["model"],
        #     re_formula="~1"
        # ).fit()
        # print(mixed.summary())
        
    
    #--------------------------
    # plot eval acc - fb32_s = 1.0 ..reducing fb21 in biased models
    #--------------------------
    df_ex = df[(df['stim_prob']=='Biased') &
               (df['fb32_scalar']==1.0)]
    # Set plot aesthetics
    sns.set(style="ticks", context="talk")
    # Initialize FacetGrid
    g = sns.FacetGrid(df_ex, col="layer", col_wrap=3, sharey=True, height = 4, aspect = 1.2)
    discrete_palette = sns.color_palette('viridis', n_colors=n_hues)
    # Add custom error bars
    hue_order = list(np.unique(df['fb21_scalar']))
    x_order = list(sorted(df_ex['cue_on'].unique(), reverse=True))
    n_hues = len(np.unique(df['fb21_scalar']))
    bar_width = 0.8
    width_per_bar = bar_width / n_hues
    # Map barplot onto each facet
    g.map_dataframe(
        sns.barplot,
        x="cue_on",
        y="eval_acc",
        hue="fb21_scalar",
        palette = discrete_palette,
        ci=None,
        errorbar=None,
        estimator=np.mean,
        dodge = True,order=x_order,
        hue_order=hue_order
    )
       
    for ax, layer in zip(g.axes.flat, sorted(df_ex['layer'].unique(), key=int)):
        subset = df_ex[df_ex['layer'] == layer]
        means = subset.groupby(['cue_on', 'fb21_scalar'])['eval_acc'].mean().reset_index()
        errors = subset.groupby(['cue_on', 'fb21_scalar'])['eval_acc'].apply(sem).reset_index()
    
        for i, row in means.iterrows():
            prob = row['cue_on']
            noise = row['fb21_scalar']
            mean = row['eval_acc']
            err = errors.loc[
                (errors['cue_on'] == prob) &
                (errors['fb21_scalar'] == noise),
                'eval_acc'
            ].values[0]
            xloc = x_order.index(prob)
            hloc = hue_order.index(noise)
            bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
            ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)
    
    # Final plot cleanup
    #g.set(ylim=(0, 3.5))
    g.set_axis_labels("", "Eval accuracy")
    g.set_titles("Layer {col_name}")
    g.add_legend(title='fb21_scalar', bbox_to_anchor=(0.86, 0.5), loc='center left')
    
    # Center shared x-axis label
    plt.subplots_adjust(bottom=0.2, left=0.12)
    g.fig.text(0.5, 0.05, 'Reducing feedback from layer 2 to 1 only', ha='center', fontsize=14)
    #g.savefig(f"decode_data/plots/eval_acc_{classes}_stimprob_x_cueon_cuelayer3_feedback21.png", format="png", bbox_inches="tight")
    plt.show()
    
    # stats

    
    
    #--------------------------
    # plot AUC - fb32_s = 1.0 ..reducing fb21 in biased models
    #--------------------------
    for noises in df['stim_noise'].unique():
        df_ex = df[(df['stim_prob']=='Biased') &
                   (df['fb32_scalar']==1.0) & (df['stim_noise']==noises)]
        # Set plot aesthetics
        sns.set(style="ticks", context="talk")
        # Initialize FacetGrid
        g = sns.FacetGrid(df_ex, col="layer", col_wrap=3, sharey=True, height = 4, aspect = 1.2)
        discrete_palette = sns.color_palette('viridis', n_colors=n_hues)
        # Add custom error bars
        hue_order = list(np.unique(df['fb21_scalar']))
        x_order = list(sorted(df_ex['cue_on'].unique(), reverse=True))
        n_hues = len(np.unique(df['fb21_scalar']))
        bar_width = 0.8
        width_per_bar = bar_width / n_hues
        # Map barplot onto each facet
        g.map_dataframe(
            sns.barplot,
            x="cue_on",
            y="delta_AUC",
            hue="fb21_scalar",
            palette = discrete_palette,
            errorbar=None,
            estimator=np.mean,
            dodge = True,order=x_order,
            hue_order=hue_order
        )
           
        for ax, layer in zip(g.axes.flat, sorted(df_ex['layer'].unique(), key=int)):
            subset = df_ex[df_ex['layer'] == layer]
            means = subset.groupby(['cue_on', 'fb21_scalar'], observed = True)['delta_AUC'].mean().reset_index()
            errors = subset.groupby(['cue_on', 'fb21_scalar'], observed = True)['delta_AUC'].apply(sem).reset_index()
        
            for i, row in means.iterrows():
                prob = row['cue_on']
                noise = row['fb21_scalar']
                mean = row['delta_AUC']
                err = errors.loc[
                    (errors['cue_on'] == prob) &
                    (errors['fb21_scalar'] == noise),
                    'delta_AUC'
                ].values[0]
                xloc = x_order.index(prob)
                hloc = hue_order.index(noise)
                bar_center = xloc - bar_width / 2 + width_per_bar / 2 + hloc * width_per_bar
                ax.errorbar(x=bar_center, y=mean, yerr=err, fmt='none', c='black', capsize=5)
        
        # Final plot cleanup
        g.set(ylim=(0, 40))
        g.set_axis_labels("", "AUC Expected - Unexpected")
        g.set_titles("Layer {col_name}")
        g.add_legend(title='Feedback 2->1', bbox_to_anchor=(0.89, 0.5), loc='center left')
        
        # Center shared x-axis label
        plt.subplots_adjust(bottom=0.2, left=0.12)
        g.fig.text(0.5, 0.05, f'Stimulus Noise {noises}', ha='center', fontsize=14)
        g.savefig(f"decode_data/plots/D_AUC_{classes}_cueon_cuelayer3_feedback21_stimnoise{int(noises*100)}.png", format="png", bbox_inches="tight")
        g.savefig(f"decode_data/plots/D_AUC_{classes}_cueon_cuelayer3_feedback21_stimnoise{int(noises*100)}.svg", format="svg", bbox_inches="tight")
        plt.show()
    
    # stats
    
    
    #--------------------------
    # plot eval acc - heatmap: interaction between fb21 and fb32
    #--------------------------
    
    df_ex = df[df['stim_prob'] == "Biased"].copy()
    df_ex["fb21_scalar"] = pd.to_numeric(df_ex["fb21_scalar"], errors="coerce")
    df_ex["fb32_scalar"] = pd.to_numeric(df_ex["fb32_scalar"], errors="coerce")
    df_ex["eval_acc"]  = pd.to_numeric(df_ex["eval_acc"], errors="coerce")
    
    # set up facet grid
    g = sns.FacetGrid(df_ex, row="layer", col="cue_on", margin_titles=True)
    
    def facet_heatmap(data, color, **kws):
        # pivot within each facet's subset
        pivoted = data.pivot_table(
            index="fb32_scalar",
            columns="fb21_scalar",
            values="eval_acc",
            aggfunc="mean"
        )
        sns.heatmap(
            pivoted,
            annot=True,
            cmap="magma",
            cbar=False,   # remove duplicate colorbars
            **kws
        )
    
    # map each facet
    g.map_dataframe(facet_heatmap)
    
    # add a single colorbar
    # (get first heatmap's mappable)
    for ax in g.axes.flat:
        im = ax.collections[0]
        break
    #g.fig.colorbar(im, ax=g.axes, orientation="vertical", pad = 0.02,shrink=0.6, fraction = .05)
    g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.3)

    plt.show()
    
    
    #--------------------------
    # plot auc - heatmap: interaction between fb21 and fb32
    #--------------------------
    
    df_ex = df[(df['stim_prob']=='Biased') &
               (df['stim_noise']==0.6)].copy()
    df_ex["fb21_scalar"] = pd.to_numeric(df_ex["fb21_scalar"], errors="coerce")
    df_ex["fb32_scalar"] = pd.to_numeric(df_ex["fb32_scalar"], errors="coerce")
    df_ex["delta_AUC"]  = pd.to_numeric(df_ex["delta_AUC"], errors="coerce")
    
    # set up facet grid
    g = sns.FacetGrid(df_ex, row="layer", col="cue_on", margin_titles=True)
    
    def facet_heatmap(data, color, **kws):
        # pivot within each facet's subset
        pivoted = data.pivot_table(
            index="fb32_scalar",
            columns="fb21_scalar",
            values="delta_AUC",
            aggfunc="mean"
        )
        sns.heatmap(
            pivoted,
            annot=True,
            cmap="magma",
            cbar=False,   # remove duplicate colorbars
            **kws
        )
    
    # map each facet
    g.map_dataframe(facet_heatmap)
    
    # add a single colorbar
    # (get first heatmap's mappable)
    for ax in g.axes.flat:
        im = ax.collections[0]
        break
    #g.fig.colorbar(im, ax=g.axes, orientation="vertical", pad = 0.02,shrink=0.6, fraction = .05)
    g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
    g.savefig(f"decode_data/plots/D_AUC_{classes}_stimprob_x_cueon_cuelayer3_heatmap_stimnoise60.png", format="png", bbox_inches="tight")
    g.savefig(f"decode_data/plots/D_AUC_{classes}_stimprob_x_cueon_cuelayer3_heatmap_stimnoise60.svg", format="svg", bbox_inches="tight")
    plt.show()
    
    # stats
    aovrm = AnovaRM(df_ex, depvar='delta_AUC', subject='model', within=['cue_on','fb32_scalar', 'fb21_scalar', 'layer'])
    res = aovrm.fit()
    print(res)
    
    
    #--------------------------
    # plot eval  - grouped bar plots
    #--------------------------
    df_ex = df[(df['stim_prob']=='Biased') &
               (df['stim_noise']==0.1)].copy()
    # make sure scalars are numeric
    df_ex["fb21_scalar"] = pd.to_numeric(df_ex["fb21_scalar"], errors="coerce")
    df_ex["fb32_scalar"] = pd.to_numeric(df_ex["fb32_scalar"], errors="coerce")
    df_ex["eval_acc"]   = pd.to_numeric(df_ex["eval_acc"], errors="coerce")
    
    # convert scalars to categorical with sorted order (so barplots look clean)
    df_ex["fb21_scalar"] = pd.Categorical(df_ex["fb21_scalar"], ordered=True, categories=sorted(df_ex["fb21_scalar"].unique()))
    df_ex["fb32_scalar"] = pd.Categorical(df_ex["fb32_scalar"], ordered=True, categories=sorted(df_ex["fb32_scalar"].unique()))
    
    # grouped barplot
    g = sns.catplot(
        data=df_ex,
        x="fb21_scalar",
        y="eval_acc",
        hue="fb32_scalar",
        col="cue_on",
        row="layer",
        kind="bar",
        palette = discrete_palette,
        margin_titles=True,
        height=4,
        aspect=1.2
    )
    
    g.set_axis_labels("fb21_scalar", "Eval Accuracy")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.add_legend(title="fb32_scalar")
    #g.savefig(f"decode_data/plots/Eval_acc_{classes}_stimprob_x_cueon_cuelayer3_grouped.png", format="png", bbox_inches="tight")
    plt.show()
    
    #--------------------------
    # plot auc  - grouped bar plots
    #--------------------------
    df_ex = df[(df['stim_prob']=='Biased') &
               (df['stim_noise']==0.1)].copy()
    # make sure scalars are numeric
    df_ex["fb21_scalar"] = pd.to_numeric(df_ex["fb21_scalar"], errors="coerce")
    df_ex["fb32_scalar"] = pd.to_numeric(df_ex["fb32_scalar"], errors="coerce")
    df_ex["delta_AUC"]   = pd.to_numeric(df_ex["delta_AUC"], errors="coerce")
    
    # convert scalars to categorical with sorted order (so barplots look clean)
    df_ex["fb21_scalar"] = pd.Categorical(df_ex["fb21_scalar"], ordered=True, categories=sorted(df_ex["fb21_scalar"].unique()))
    df_ex["fb32_scalar"] = pd.Categorical(df_ex["fb32_scalar"], ordered=True, categories=sorted(df_ex["fb32_scalar"].unique()))
    
    # grouped barplot
    g = sns.catplot(
        data=df_ex,
        x="fb21_scalar",
        y="delta_AUC",
        hue="fb32_scalar",
        col="cue_on",
        row="layer",
        kind="bar",
        palette = discrete_palette,
        margin_titles=True,
        height=4,
        aspect=1.2
    )
    
    g.set_axis_labels("fb21_scalar", "Δ AUC")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.add_legend(title="fb32_scalar")
    #g.savefig(f"decode_data/plots/D_AUC_{classes}_stimprob_x_cueon_cuelayer3_grouped.png", format="png", bbox_inches="tight")
    plt.show()
   


# at the end remind me which one we were working on  
print('\007') # make a sound   
print(f'finished {settings}')



