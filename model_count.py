#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:10:40 2025

@author: hkular
"""

import os
from collections import Counter

def count_all_models(task_type):
    # path to your directory
    directory = os.getcwd()
    
    counts = Counter()
    
    if task_type == 'rdk':
        fulldir = f'{directory}/trained_models_rdk'
    elif task_type == 'rdk_reproduction':
        fulldir = f'{directory}/trained_models_rdk_reproduction'
    elif task_type == 'rdk_repro_cue':
        fulldir = f'{directory}/trained_models_rdk_repro_cue'
    for filename in os.listdir(fulldir):
        if filename.endswith('.pt'):
            model_name = filename.split('modnum-')[0].rstrip('_')  # everything before "modnum-", remove trailing '_'
            counts[model_name] += 1
    
    # Now 'counts' has the counts for each unique model
    for model, count in counts.items():
        print(f'{model}: {count} modnums')





def count_models( n_afc, stim_prob, stim_amps, stim_noise, weighted_loss, task_type, fn_stem, directory):
    
    if not directory:
        directory = os.getcwd()
        print(f'Looking at models in directory: {directory}')
        
    if task_type == 'rdk_reproduction':
        if weighted_loss == 0:
            fn = f'repro_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1_nw_mse'
        else:
            # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
            if stim_prob == 1 / n_afc:
                #fn = f'trained_models/num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_nw_mse_modnum-{m_num}.pt'
                fn = f'repro_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1'
            else:
                fn = f'repro_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1'  
        
        # define full path to emodel directory
        model_dir = os.path.join(directory, 'trained_models_rdk_reproduction')
        if not os.path.exists(model_dir):
            print(f"Directory {model_dir} does not exist.")
            return 0
        
       # Count files starting with the prefix
        count = sum(1 for fname in os.listdir(model_dir) if fname.startswith(fn) and fname.endswith('.pt'))
        print(f"Found {count} models matching prefix '{fn}'")
        return count
    elif task_type == 'rdk_repro_cue':
        if weighted_loss == 0:
            fn = f'reprocue_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1_nw_mse'
        else:
            # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
            if stim_prob == 1 / n_afc:
                #fn = f'trained_models/num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_nw_mse_modnum-{m_num}.pt'
                fn = f'reprocue_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1'
            else:
                fn = f'reprocue_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1'  
        
        # define full path to emodel directory
        model_dir = os.path.join(directory, fn_stem[:-9])
        if not os.path.exists(model_dir):
            print(f"Directory {model_dir} does not exist.")
            return 0
       # Count files starting with the prefix
        count = sum(1 for fname in os.listdir(model_dir) if fname.startswith(fn) and fname.endswith('.pt'))
        print(f"Found {count} models matching prefix '{fn}'")
        return count
    elif task_type == 'rdk':
        if weighted_loss == 0:
            fn = f'gonogo_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1_nw_mse'
        else:
            # if equal prob, then loss already evenly weighted across stims so can use the "nw_mse" version (non-weighted mse loss)
            if stim_prob == 1 / n_afc:
                #fn = f'trained_models/num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_nw_mse_modnum-{m_num}.pt'
                fn = f'gonogo_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1'
            else:
                fn = f'gonogo_num_afc-{n_afc}_stim_prob-{int( stim_prob * 100 )}_stim_amp-{int( stim_amps * 100 )}_stim_noise-{int( stim_noise * 100 )}_h_bias_trainable-1'  
        
        # define full path to emodel directory
        model_dir = os.path.join(directory, 'trained_models_rdk')
        if not os.path.exists(model_dir):
            print(f"Directory {model_dir} does not exist.")
            return 0
        
       # Count files starting with the prefix
        count = sum(1 for fname in os.listdir(model_dir) if fname.startswith(fn) and fname.endswith('.pt'))
        print(f"Found {count} models matching prefix '{fn}'")
        return count

    print(f"Unsupported task_type: {task_type}")
    return 0

        

# In[]
#### Code for making plots of input and outputs

# import matplotlib.pyplot as plt
# # plot example outputs


# plt.plot(outputs[:,5,0])
# plt.show()



# # if outputs were nafc

# base = outputs.squeeze(axis=2)  # Now shape is (200,3000)

# # Now, expand it by adding 0, 1, 2, 3, 4, 5
# expanded = np.stack([base + n for n in range(6)], axis=-1)



# sample = expanded[:, 0, :]  # shape (3000, 6)


# for i in range(6):
#     plt.plot(sample[:, i], label=f'Channel {i}')

# plt.xlabel('Time')
# plt.ylabel('Output')
# plt.show()



# # Look at trials
# idx = s_label == 1
# plt.figure()
# first_else_encountered = False 
# for i in range(10):
#     if idx[i]:
#         plt.plot(outputs[:,i, 0], hex_c[1])
#     else:
#         #plt.plot(outs[i, :], 'r', label='Stim 1' if i < 1 else '_nolegend_')
#         if not first_else_encountered:
#             plt.plot(outputs[:,i, 0], c=hex_c[0], label='Expected')
#             first_else_encountered = True  # Update flag
#         else:
#             plt.plot(outputs[:,i, 0], c= hex_c[0], label='_nolegend_')
        
# plt.xlabel('Time Steps')
# plt.ylabel('Output (au)')
# plt.title('Output Signals from 100 Trials')
# plt.legend()
# plt.show()




# plt.figure()
# plt.plot(outputs[:,5, 0], c=hex_c[0], label='Expected')
# plt.plot(outputs[:,1, 0], c=hex_c[1], label='Unexpected')
# plt.xlabel('Time Steps')
# plt.ylabel('Output (au)')
# plt.legend()
# plt.show()



# inputs,s_label = task.generate_rdk_stim() 
# plt.plot(inputs[:,9,:])
# plt.xlabel('Time Steps')
# plt.ylabel('Inputs (au)')
# plt.show()



# targets = task.generate_rdk_target( s_label )

# plt.plot(targets[:,5])
