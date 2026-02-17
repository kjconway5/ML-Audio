# Model Training  

1. First change config.yaml to adjust: 
  -Parameters: 
    -Log-Mel Parameters: FFT size, n_mels, sample rate...
    -DS-CNN config: # of filters, batch size, kernel size, stride... 
    -Target keywords/classes ("yes", "no", "silence"), n_classes must match number of target 
      keywords + silence + unkown 
    -Output files (.pt is where model is saved)

2. Pre-process data 
  python process_data.py
 
3. Train model 
  python train.py 

4. Test Model 
  Record from microphone:
  python test_live_audio.py -m model_final.pt

  Test a WAV file:
  python test_live_audio.py -m model_final.pt -f audio.wav

  List audio devices:
  python test_live_audio.py --list-devices

Requirements

  pip install torch torchaudio pyyaml numpy scipy soundfile tqdm sounddevice

  Python 3.10.13                                                                                                                 
                                                                                                                        
  torch==1.12.1+cu102
  torchaudio==0.12.1+cu102
  numpy==1.26.4
  scipy==1.15.3
  soundfile==0.13.1
  sounddevice==0.5.5
  tqdm==4.67.3
  pyyaml==6.0.3
  cffi==2.0.0