DS-CNN specifics README.md

PyTorch Model Structure: 
dscnn.py - 
  - First conv: A large (10×4) strided convolution that takes the 1-channel mel spectrogram and produces 24 feature maps, using stride (2,2) to  downsample the spatial dimensions.

  - DS blocks: Four sequential blocks each consisting of a depthwise conv (one filter per channel) followed by a pointwise 1×1 conv (mixing information across channels). Key Idea: DS-CNN vs tiny-CNN reduces number of parameters and computations ~6.5x

Trained Models: 
dscnn7.pt - Trained with old preprocessing pipeline that did not simulate RTL functionality 
dscnn7-new.pt - Trained with NEW preprocessing pipeline that simulates RTL functionality 
dscnn-golden.pt - Trained with golden_model.py (entire pipeline in one file )


####
config.yaml for all ds-cnn: 

dataset:
  data_dir: "/home/dnocera/ML-Audio/ml/data/speech_commands"

data:
  output_dir: "/home/dnocera/ML-Audio/ml/output"

  classes:
    - "yes"
    - "no"
    - "left"
    - "right"
    - "go"
    - "silence"
    - "unknown"

  target_keywords:
    - "yes"
    - "no"
    - "go"
    - "left"
    - "right"

  include_silence: true
  num_silence_samples: 1000
  num_true_silence_samples: 500
  unknown_max_per_split: 4000
  random_seed: 42


preprocessing:
  sample_rate: 16000
  n_mels: 40
  n_fft: 256
  hop_length: 128
  window_length: 256

  #filters
  use_filters: false
  hpf_order: 2
  lpf_order: 4
  cutoff_hpf: 150
  cutoff_lpf: 4000

#ds-cnn
model:
  n_classes: 7  # yes, no, stop, go, silence, unknown
  
  # First convolution layer
  first_conv:
    filters: 24
    kernel_size: [10, 4]
    stride: [2, 2]
  
  # Depthwise Separable blocks
  ds_blocks:
    n_blocks: 4
    filters: 24
    kernel_size: [3, 3]
    stride: [1, 1]

training:
  n_epochs: 20
  batch_size: 64

  optimizer: "sgd"
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 0.0001

  lr_schedule:
    milestones: [5, 10, 15]
    gamma: 0.1
  val_every: 5

vad:
  calibration_duration: 2.0
  threshold_multiplier: 3.0
  default_energy_threshold: 0.0001





