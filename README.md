# Design (updated 080825)

## Train models with following parameters
- Timing
  - T = 210 (increased from 200 to avoid square matrices)
  - Cue_on = stim_offset (75) or Cue_on = 0
  - Stim_on = 50
  - Stim_dur = 25
- Task
  - rdk_repro_cue: produce 1 or 0 in n_stim output channels and get cued to scramble stim->resp mapping
  - n_afc = 6
  - stim_amps, stim_noise, and model_noise all on easiest setting for now
  - stim_prob = unbiased (1/n_afc) or biased (70% stim0 and 30% others)
  - cue location: h layer 3 (can do others), cue goes to all units
- Model
  - 3 hidden layers
  - trained to 90% accuracy or loss <0.001 (decreased training accuracy to increase incorrect trials at eval)
  - training batch size = 256 (should be sufficient for decrease in accuracy)
  - loss function is unweighted (weighted mse is an option we decided against because of human expt prioritizing expected stims)

## Analyses Plans
- LS-SVM decoding accuracy
  - compare cue 0 and cue 75
  - compare correct and incorrect trials
  - decode cue identity (stim identity default for other analyses)
  - compare xgen for excitatory and inhibitory units
- CEBRA: will take a look with only time structure as auxiliary vars

# Instructions (updated 080125)

## Environment

python = 3.12.2 <br />
pytorch = 2.5.1 <br />
conda env create -f environment.yml <br />
conda env activate torch_rnn


## Training

python run_model.py --N 10 --gpu 1 --task rdk_repro_cue --n_afc 6 --int_noise 0.1 --ext_noise 0.1  --n_cues 2 --stim_amps 1.0 --stim_prob_mode biased



## Evaluation

If you want lots of plots to look at eval performance and decoding accuracy within a single layer:

* decode_single_layer.py -- you will need to specify which type of model and what layer


If you want to compare decoding accuracy across layers in a single plot:

* decode_multilayer.py

