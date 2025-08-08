#!/usr/bin/env python
# coding: utf-8

# Name: Holly Kular\
# Date: 10-18-2024\
# Email: hkular@ucsd.edu\
# Description: look at PCA of stim representations in RNN over time

# In[14]:


import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
from scipy.stats import ttest_ind
from scipy.spatial.distance import euclidean
from itertools import combinations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from helper_funcs import *
from rdk_task import RDKtask
from model_count import count_models


# In[2]:

#--------------------------
# Basic model params
#--------------------------
device = 'cpu'                    # device to use for loading/eval model
task_type = 'rdk_repro_cue'    # task type (conceptually think of as a motion discrimination task...)         
n_afc = 6                         # number of stimulus alternatives
T = 225                           # timesteps in each trial
stim_on = 50                      # timestep of stimulus onset
stim_dur = 25                     # stim duration
stim_prob = 0.8               # probability of stim 1, with probability of (1-stim_prob)/(n_afc-1) for all other options
stim_prob_to_eval = 1/n_afc     # eval the model at this prob level (stim_prob is used to determine which trained model to use)
stim_amps_train = 1.0            # can make this a list of amps and loop over... 
stim_amps = 1.0
stim_noise_train = 0.1
stim_noise = 0.1                  # magnitude of randn background noise in the stim channel for eval
batch_size = 1000                 # number of trials in each batch
acc_amp_thresh = 0.8              # to determine acc of model output: > acc_amp_thresh during target window is correct
weighted_loss = 0                 #  0 = nw_mse l2 or 1 = weighted mse
noise_internal = 0.1              # trained under 0.1 try 0.25 
num_cues = 2
cue_on = stim_on+stim_dur
cue_dur = T-cue_on
cue_layer = 3
out_size = n_afc  
fn_stem = f'trained_models_rdk_repro_cue/timing_{T}_cueon_{cue_on}/cue_layer{cue_layer}/reprocue_'

    


#--------------------------
# decoding params
#--------------------------
trn_prcnt = 0.8    # percent of data to use for training
n_cvs = 5  # how many train/test cv folds
time_or_xgen = 0   # decode timepnt x timepnt or do full xgen matrix 
w_size = 5         # mv_avg window size
num_cs = 1         # number of C's to grid search, if 1 then C=1
n_cvs_for_grid = 5 # num cv folds of training data to find best C
max_iter = 5000    # max iterations

#--------------------------
# init dict of task related params
# note that stim_prob_to_eval is passed in here
# and that the model name (fn) will be generated based 
# on stim_prob... 
#--------------------------
settings = {'task' : task_type, 'n_afc' : n_afc, 'T' : T, 'stim_on' : stim_on, 'stim_dur' : stim_dur,
            'stim_prob' : stim_prob_to_eval, 'stim_amp' : stim_amps, 'stim_noise' : stim_noise, 'batch_size' : batch_size, 
            'acc_amp_thresh' : acc_amp_thresh, 'out_size' : out_size, 'num_cues':num_cues, 'cue_on':cue_on, 'cue_dur':cue_dur}

# create the task object
task = RDKtask( settings )


# just pick first model for now
m_num = 0
    
# build a file name...
if weighted_loss == 0:
    fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_nw_mse_modnum-{m_num}.pt'
else:
    # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
    if stim_prob == 1 / n_afc:
        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'
    else:
        fn = f'{fn_stem}num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps_train * 100 )}_stim_noise-{int( stim_noise_train * 100 )}_h_bias_trainable-1_modnum-{m_num}.pt'  

    
# load the trained model, set to eval, requires_grad == False
net = load_model( fn,device )
# load cue scramble matrix
with open(f'{fn[:-3]}.json', "r") as infile:
   _ , rnn_settings = json.load(infile)
sr_scram_list = rnn_settings['sr_scram']
sr_scram_list = [sr_scram_list[str(s)] for s in sorted(sr_scram_list.keys(), key=int)]
sr_scram = np.array(sr_scram_list)


print(f'loaded model {m_num}')

# update eval noise to bring class acc off ceiling
net.recurrent_layer.noise = noise_internal

# eval a batch of trials using the trained model
outputs,s_label,h1,h2,h3,ff12,ff23,fb21,fb32,tau1,tau2,tau3,m_acc,tbt_acc, cues = eval_model( net, task, sr_scram )

# s_label is a diff shape for cue version, deal with if statement later
s_label_int = np.argmax(s_label, axis=1)
unique_labels = np.unique(s_label_int)

# %%


# ### labels in one plot clouds of start and end points

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

fr = h3
labels = s_label_int

# Define a colormap for the labels
colormap = colormaps.get_cmap('tab10')
colors = [colormap(i / len(unique_labels)) for i in range(len(unique_labels))]


for i, label in enumerate(unique_labels):

    # Filter trials for this label
    label_idx = labels == label
    fr_label = fr[:,label_idx,:]  # trials for this label, shape: trials x time x units

    # Reshape trials into time-units
    fr_col = fr_label.reshape((fr_label.shape[0] * fr_label.shape[1], fr_label.shape[2]))

    # Perform PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(fr_col)  # shape: (total_points, 3)

    # Reshape to identify time points per trial
    pcs_r = pcs.reshape((fr_label.shape[0], fr_label.shape[1], 3))  # (time,trials, components)

    # Flatten data for plotting all points
    pcs_flat = pcs_r.reshape((-1, 3))  # shape: (trials * time, 3)

    # Scatter all points
    #ax.scatter(pcs_flat[:, 0], pcs_flat[:, 1], pcs_flat[:, 2], alpha=0.5, label=f'Label {label}')

    # Mark the start and end points
    ax.scatter(pcs_r[:, 0, 0], pcs_r[:, 0, 1], pcs_r[:, 0, 2], label = f'stim {label}', color=colors[i]) 
    ax.scatter(pcs_r[0, 0, 0], pcs_r[0, 0, 1], pcs_r[0, 0, 2],  color='green', s =100) # First time point
    ax.scatter(pcs_r[-1, 0, 0], pcs_r[-1, 0, 1], pcs_r[-1, 0, 2], color='red', s =100)  # Last time point

    # Set labels and title
    #ax.set_title(f'Label {label}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# %%


# ### Compare start and end points - stats


# Prepare arrays to store results
start_points = []
end_points = []
labels_all = []
fr = h1 # shape (1000, 250, 200) trials x time x units
labels = s_label_int  # labels are 1D, shape (1000,)trials



for label in unique_labels:
    # Filter trials and calculate mean trajectory
    label_idx = labels == label
    fr_label = fr[label_idx]
    fr_col = fr_label.reshape((fr_label.shape[0] * fr_label.shape[1], fr_label.shape[2]))
    pcs = PCA(n_components=3).fit_transform(fr_col)
    pcs_r = pcs.reshape((fr_label.shape[0], fr_label.shape[1], 3))
    #pcs_s = pcs_r.mean(axis=0)
    
    # Extract start and end points
    start_points.append(pcs_r[:, 0, :])  # Start point
    end_points.append(pcs_r[:, -1, :])  # End point

start_points = np.array(start_points)  # shape: (n_labels,)
end_points = np.array(end_points)      # shape: (n_labels,)


# Perform pairwise comparisons between labels
results = []
for (label1, label2) in combinations(range(len(unique_labels)), 2):
    # Compute Euclidean distances between start points of label1 and label2
    t_stat_start, p_val_start = ttest_ind(
        start_points[label1],
        start_points[label2],
        equal_var=False, axis = 0
    )
    # Compute Euclidean distances between end points of label1 and label2
    t_stat_end, p_val_end = ttest_ind(
        end_points[label1],
        end_points[label2],
        equal_var=False, axis = 0
    )
    
    results.append({
        "Labels": (unique_labels[label1], unique_labels[label2]),
        "Start p-value": p_val_start,
        "End p-value": p_val_end,
    })


# In[36]:


for result in results:
    start_pvals = result["Start p-value"]  # Should be a list or tuple of 3 p-values
    end_pvals = result["End p-value"]      # Same as above
    labels = result["Labels"]

    # Check if any p-value in the start or end exceeds 0.01
    if any(p < 0.01 for p in start_pvals) or any(p < 0.01 for p in end_pvals):
        print(f"Labels: {labels}")
        print(f"Start p-values: {start_pvals}")
        print(f"End p-values: {end_pvals}")


# ### calculate PCA and save plot for each layer - all labels in one plot

# In[4]:


# Initialize the figure

for layer in range(3):

    layer+=1
    fr = data[f'fr{layer}']
    labels = data['labs'].flatten() 

    unique_labels = np.unique(labels)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define a colormap for the labels
    colormap = colormaps.get_cmap('tab10')
    colors = [colormap(i / len(unique_labels)) for i in range(len(unique_labels))]

    for i, label in enumerate(unique_labels):
        # Filter trials for this label
        label_idx = labels == label
        fr_label = fr[label_idx]  # trials for this label, shape: trials x time x units

        # Average across trials to simplify
        #fr_mean = fr_label.mean(axis=0)  # shape: time x units
        # collapse across trials and time
        fr_col = fr_label.reshape((fr_label.shape[0]*fr_label.shape[1], fr_label.shape[2]))

        # Perform PCA
        pca = PCA(n_components=3)
        pcs_s = pca.fit_transform(fr_col)  # shape: time x components
        pcs_r = pcs_s.reshape((fr_label.shape[0], fr_label.shape[1], 3))
        pcs = pcs_r.mean(axis=0)

        # Plot trajectory
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], label=f'Stim {label}', color=colors[i], alpha=0.7, linewidth = 2)

        # Mark start (green) and end (red) points
        ax.scatter(pcs[0, 0], pcs[0, 1], pcs[0, 2], color='green', s=50, label= '_nolegend_')
        ax.scatter(pcs[-1, 0], pcs[-1, 1], pcs[-1, 2], color='red', s=50, label='_nolegend_')

    # Add axis labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Add title and legend
    ax.set_title('3D PCA Trajectories for All Stims')
    ax.legend()
    plt.tight_layout()
    fname = f"/Volumes/serenceslab/holly/RNN_Geo/04_PCA/{afc}afc/PCA_stims_fr{layer}_{coh}_mod{mod}"
    #plt.savefig(fname)
    plt.show()


# # calculate the multidimensional euclidean distance

# #### plot euclid distances for single model all layers

# In[72]:


# calculate distances for each layer and create plot


labels = data['labs'].flatten() 
unique_labels = np.unique(labels)

# Initialize a figure
plt.figure(figsize=(12, 6))

# Color map for different layers
layer_colormap = plt.cm.get_cmap('Set1')
colormap = colormaps.get_cmap('tab10')

# Loop over layers
for layer_idx, layer in enumerate([1]):  
    
    fr = data[f'fr{layer}']

    # Define a colormap for the labels
    colors = [colormap(i / len(unique_labels)) for i in range(len(unique_labels))]

    pcs_all = np.zeros((fr.shape[1], 3, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        # Filter trials for this label
        label_idx = labels == label
        fr_label = fr[label_idx]  # trials for this label, shape: trials x time x units

        # collapse across trials and time
        fr_col = fr_label.reshape((fr_label.shape[0]*fr_label.shape[1], fr_label.shape[2]))

        # Perform PCA
        pca = PCA(n_components=3)
        pcs_s = pca.fit_transform(fr_col)  # shape: time x components
        pcs_r = pcs_s.reshape((fr_label.shape[0], fr_label.shape[1], 3)) # uncollapse trials x time
        pcs_all[:,:, i] = pcs_r.mean(axis=0) #store for each label


    # distances
    num_pairs = len(unique_labels) * (len(unique_labels) - 1) // 2  # Number of unique pairs
    distances = np.zeros((fr.shape[1], num_pairs))
    pair_idx = 0
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            # pairwise distance for each time point - L2 norm which is euclidean norm
            distances[:, pair_idx] = np.linalg.norm(
                pcs_all[:, :, i] - pcs_all[:, :, j], axis=1
            )
            pair_idx += 1      

# Plot distances for this layer with a unique color
    layer_color = layer_colormap(layer_idx / 3)
    for pair in range(num_pairs):
        if pair > 4:
            color_pair = "red"
        else: color_pair = "blue"
        plt.plot(np.arange(pcs_all.shape[0]), distances[:, pair], 
                 color=color_pair, 
                 label=f"Layer {layer} - Pair {pair+1}", 
                 alpha=0.7)  # Added some transparency

# Add stimulus period
stim_on = 80
stim_off = 80+50
plt.axvspan(stim_on, stim_off, color='gray', alpha=0.3, label="Stimulus On")

plt.xlabel("Time Steps")
plt.ylabel("Euclidean Distance")
plt.title("Euclidean Distances Across Layers")
plt.legend()
plt.tight_layout()
#fname = f"/Volumes/serenceslab/holly/RNN_Geo/04_PCA/{afc}afc/PCA_dists_allfrs_{coh}_mod{mod}"
#plt.savefig(fname)
plt.show()


# #### plot euclid distances for all models all layers

# In[51]:


# List of model numbers to loop over
models_to_process = [0, 1, 2]

# Initialize a figure
plt.figure(figsize=(15, 8))

# Define consistent layer colors
layer_colors = {
    1: 'red',     # Layer 1 will be in reds
    2: 'green',  # Layer 2 will be in oranges
    3: 'blue'     # Layer 3 will be in blues
}

# Define line styles for models
line_styles = ['-', '--', '-.']

# Function to generate color variations for a base color
def generate_color_variations(base_color, num_variations):
    # Create a colormap with variations of the base color
    cmap = plt.cm.get_cmap('Reds') if base_color == 'red' else            plt.cm.get_cmap('Greens') if base_color == 'green' else            plt.cm.get_cmap('Blues')
    return [cmap(i/num_variations) for i in range(1, num_variations+1)]

# Loop over layers first to maintain color consistency
for layer_idx, layer in enumerate([1, 2, 3]):
    # Generate color variations for this layer
    model_layer_colors = generate_color_variations(layer_colors[layer], len(models_to_process))
    
    # Loop over models
    for model_idx, mod in enumerate(models_to_process):
        # Construct the model path
        model_path = os.path.join(data_dir, mat_files[mod]) 
        model = loadmat(model_path)   
        
        # Construct the data file path
        data_file = f"{data_dir}/Trials1000_model{model_path[-7:-4]}_neutral.npz"
        data = np.load(data_file)
        
        # Load data for this layer
        fr = data[f'fr{layer}']
        labels = data['labs'].flatten() 
        unique_labels = np.unique(labels)
        
        # PCA and distance calculations (same as previous code)
        pcs_all = np.zeros((fr.shape[1], 3, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            # Filter trials for this label
            label_idx = labels == label
            fr_label = fr[label_idx]
            fr_col = fr_label.reshape((fr_label.shape[0]*fr_label.shape[1], fr_label.shape[2]))
            
            # Perform PCA
            pca = PCA(n_components=3)
            pcs_s = pca.fit_transform(fr_col)
            pcs_r = pcs_s.reshape((fr_label.shape[0], fr_label.shape[1], 3))
            pcs_all[:,:, i] = pcs_r.mean(axis=0)
        
        # distances
        num_pairs = len(unique_labels) * (len(unique_labels) - 1) // 2
        distances = np.zeros((fr.shape[1], num_pairs))
        pair_idx = 0
        for i in range(len(unique_labels)):
            for j in range(i+1, len(unique_labels)):
                distances[:, pair_idx] = np.linalg.norm(
                    pcs_all[:, :, i] - pcs_all[:, :, j], axis=1
                )
                pair_idx += 1      
        
        # Plot distances for this layer and model
        model_layer_color = model_layer_colors[model_idx]
        model_line_style = line_styles[model_idx]
        
        for pair in range(num_pairs):
            plt.plot(np.arange(pcs_all.shape[0]), distances[:, pair], 
                     color=model_layer_color, 
                     linestyle=model_line_style,
                     label=f"Model {mod} - Layer {layer} - Pair {pair+1}", 
                     alpha=0.9)

# Add stimulus period
stim_on = 80
stim_off = 80+50
plt.axvspan(stim_on, stim_off, color='gray', alpha=0.3, label="Stimulus On")

plt.xlabel("Time Steps")
plt.ylabel("Euclidean Distance")
plt.title("Euclidean Distances Across Models and Layers")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fname = f"/Volumes/serenceslab/holly/RNN_Geo/04_PCA/{afc}afc/PCA_dists_allfrs_{coh}_allmods"
plt.savefig(fname)
plt.show()


# #### plot euclid distances avg across models all layers

# In[58]:


# List of model numbers to loop over
models_to_process = [0, 1, 2]

# Initialize a figure
plt.figure(figsize=(15, 8))

# Define consistent layer colors
layer_colors = {
    1: 'red',     # Layer 1 will be in reds
    2: 'orange',  # Layer 2 will be in oranges
    3: 'blue'     # Layer 3 will be in blues
}

# Line styles for individual model lines
line_styles = ['-', '--', '-.']

# Initialize dictionary to store all distances for averaging
layer_distances = {
    1: [],
    2: [],
    3: []
}

# Loop over layers first
for layer_idx, layer in enumerate([1, 2, 3]):
    # Temporary storage for this layer's model distances
    layer_model_distances = []
    
    # Loop over models
    for model_idx, mod in enumerate(models_to_process):
        # Construct the model path
        model_path = os.path.join(data_dir, mat_files[mod]) 
        model = loadmat(model_path)   
        
        # Construct the data file path
        data_file = f"{data_dir}/Trials1000_model{model_path[-7:-4]}_neutral.npz"
        data = np.load(data_file)
        
        # Load data for this layer
        fr = data[f'fr{layer}']
        labels = data['labs'].flatten() 
        unique_labels = np.unique(labels)
        
        # PCA and distance calculations (same as previous code)
        pcs_all = np.zeros((fr.shape[1], 3, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            # Filter trials for this label
            label_idx = labels == label
            fr_label = fr[label_idx]
            fr_col = fr_label.reshape((fr_label.shape[0]*fr_label.shape[1], fr_label.shape[2]))
            
            # Perform PCA
            pca = PCA(n_components=3)
            pcs_s = pca.fit_transform(fr_col)
            pcs_r = pcs_s.reshape((fr_label.shape[0], fr_label.shape[1], 3))
            pcs_all[:,:, i] = pcs_r.mean(axis=0)
        
        # distances
        num_pairs = len(unique_labels) * (len(unique_labels) - 1) // 2
        distances = np.zeros((fr.shape[1], num_pairs))
        pair_idx = 0
        for i in range(len(unique_labels)):
            for j in range(i+1, len(unique_labels)):
                distances[:, pair_idx] = np.linalg.norm(
                    pcs_all[:, :, i] - pcs_all[:, :, j], axis=1
                )
                pair_idx += 1      
        
        # Store model distances for this layer
        layer_model_distances.append(distances)
        
        # Plot individual model lines
        model_layer_color = plt.cm.get_cmap('Reds')(model_idx/len(models_to_process)) if layer == 1 else                             plt.cm.get_cmap('Oranges')(model_idx/len(models_to_process)) if layer == 2 else                             plt.cm.get_cmap('Blues')(model_idx/len(models_to_process))
        
        for pair in range(num_pairs):
            plt.plot(np.arange(pcs_all.shape[0]), distances[:, pair], 
                     color=model_layer_color, 
                     linestyle=line_styles[model_idx],
                     label=f"Model {mod} - Layer {layer} - Pair {pair+1}", 
                     alpha=0.4)
    
    # Calculate and store average distances for this layer
    layer_avg_distances = np.mean(layer_model_distances, axis=0)
    layer_distances[layer] = layer_avg_distances

# Plot averaged distances
for layer_idx, layer in enumerate([1, 2, 3]):
    layer_color = layer_colors[layer]
    distances = layer_distances[layer]
    num_pairs = distances.shape[1]
    
    for pair in range(num_pairs):
        plt.plot(np.arange(layer_avg_distances.shape[0]), distances[:, pair], 
                 color=layer_color, 
                 linewidth=3,  # Make average lines thicker
                 label=f"Layer {layer} - Pair {pair+1} (Avg)", 
                 alpha=0.8)

# Add stimulus period
stim_on = 80
stim_off = 80+50
plt.axvspan(stim_on, stim_off, color='gray', alpha=0.3, label="Stimulus On")

plt.xlabel("Time Steps")
plt.ylabel("Euclidean Distance")
plt.title("Averaged Euclidean Distances Across Models")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
fname = f"/Volumes/serenceslab/holly/RNN_Geo/04_PCA/{afc}afc/PCA_dists_allfrs_{coh}_avgmods"
plt.savefig(fname)
plt.show()


# In[22]:


# stats on distances

# cluster analysis
# k means to show that stim0 is in it's own cluster

# temporal analysis
# permutation test to show distances differ pre and post stimulus

# multidimensional scaling
# show stim0 is in a distinct cluster and then show separability




    


# In[ ]:




