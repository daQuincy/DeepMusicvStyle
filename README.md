# When Music Meets A.I.
[Paper](https://ieeexplore.ieee.org/document/9102870/media#media) | [Music Samples (.mp3)](https://bit.ly/3b5QYKW)

### Official TF implementation of the paper: "Style-conditioned Music Generation"

#### ICME 2020 (oral)

Released on March 18, 2020.

## Description
This is a joint work with the Department of Music (University of Malaya). It presents a refinement to the vanilla formulation of Variational Auto-Encoder (VAE) which allow users to condition the compositional style of music generated by the model. In our experiments, we trained our model on Bach chorales (JSB) and western folk tunes (NMD). At generation time, users can specify the model to generate music in the style of Bach or folk tunes. The datasets used in the experiment can be downloaded at [POP](http://www.ambrosepianotabs.com/page/library?pg=1), [JSB](http://kern.humdrum.org/search?s=t&keyword=Bach%20Johann&fbclid=IwAR39fsc8gUWjN6eYAUkewldNkeV499lX0Ew6VP8Nrrd_T1T7plaIIIb5nFQ) and [NMD](http://www-etud.iro.umontreal.ca/~boulanni/icml2012).

Curious how music generated by our model sound? Feel free to visit [Generated Music Samples](https://bit.ly/3b5QYKW) and leave your feedbacks. 

## Dependencies
- Python 3.6.8
- tensorflow(gpu) 1.15.0
- tensorflow-probability 0.8.0
- pretty-midi 0.2.8  

Tested on Ubuntu 16.04.

## Running the code
### Setup
Check _dataset.py_ in the dataset folder and put in the correct folder path for the MIDI files. Change the train/test split ratio as you desire. The output should be train/test pickle files for each style, i.e. with 2 styles, you should have 4 pickle files, train/test for style 1 and train/test for style 2.

### Training
Check _train.sh_ to tune the hyperparameters. For hyperparameters not in _train.sh_, please look into _model.py_. Those hyperparameters that are not in _train.sh_ are found to be have little impact on model's learning.

Description for hyperparameters (and values used in the paper) and options in _train.sh_:
```
save_path=vae_model                       # model save path
train_set="jsb_train.pkl nmd_train.pkl"   # training dataset path, separate style with space
test_set="jsb_test.pkl nmd_test.pkl"      # testing dataset path, make sure style sequence is same as above
epoch=400                                 # training epoch
enc_rnn=hyperlstm                         # encoder RNN: lstm, hyperlstm
enc_rnn_dim=512                           # encoder RNN dimension
enc_rnn_layer=1                           # number of encoder RNN layers
enc_hyper_unit=256                        # number of hyper units if hyperlstm is chosen
enc_dropout=0.5                           # encoder RNN dropout rate
dec_rnn=hyperlstm                         # decoder RNN: lstm, hyperlstm
dec_rnn_dim=512                           # decoder RNN dimension
dec_hyper_unit=256                        # number of hyper units if hyperlstm is chosen
dec_dropout=0.2                           # decoder RNN dropout rate
dec_rnn_layer=1                           # number of decoder RNN layers
attention=0                               # dimension for attention units for decoder self-attention (0: disable)
cont_dim=120                              # latent space size for z_c 
cat_dim=2                                 # number of styles (categorical dimension)
gumbel=0.02                               # Gumbel softmax temperature
style_embed_dim=80                        # dimension for each style embeddings in z_s
mu_force=1.3                              # beta in [mu_forcing](https://arxiv.org/abs/1905.10072)
batch_size=64                             # batch size
```

### Generating music
Run _generate.sh_ to generate music. Options are described as follows.
```
ckpt_path=vae_model/vae-epoch250          # path to saved model checkpoint
output_save_path=jsb_samples              # folder directory to save generated music
n_generations=20                          # number of music samples to generate
style=0                                   # style of music to generate, number corresponds to train_set style sequence in train.sh
output_type=all                           # output file type. midi=MIDI file, wav=WAV file, all=both
temperature=0.2                           # lower temperature gives more confident output
cont_dim=120                              # latent space size for z_c 
cat_dim=2                                 # number of styles (categorical dimension)
```

## Feedback 
Suggestions and opinions of any sort are welcomed. Please contact the authors by sending emails to `yuquan95 at gmail.com` or `cs.chan at um.edu.my` or `loofy at um.edu.my`

## License and Copyright
This project is open source under the BSD-3 license (see [`LICENSE`](./LICENSE)). Codes can be used freely only for academic purpose.

For commercial purpose usage, please contact Dr. Chee Seng Chan at `cs.chan at um.edu.my`

&#169;2020 Center of Image and Signal Processing, Faculty of Computer Science and Information Technology, University of Malaya.
