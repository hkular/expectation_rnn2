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

