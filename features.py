import torch
import numpy as np
import torchaudio.transforms as T

# Feautre extraction
class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=64, 
                 hop_length=160, window_length=512):
        """
        Args:
            sample_rate: Sample rate in Hz
            n_fft: FFT size
            n_mels: Number of mel bins
            hop_length: Samples between successive frames
            window_length: Window size (defaults to n_fft)
        """
        self.sample_rate = sample_rate # same as PCM output
        self.n_fft = n_fft # Frequency resolution is sample_rate/n_fft
        self.n_mels = n_mels # mel bands
        self.hop_length = hop_length # 160 samples at 16 KHz is 10 ms
        self.window_length = window_length or n_fft # Default 512 bc 32 ms window
                                                    # Longer window is better freq. resolution, shorter is better time resolution
        
        # create mel spectrogram transform
        # For each frame:
            # window(signal)
            # FFT
            # |FFT|^2
            # mel_filterbank
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, # convert FFT bins to Hz
            n_fft=n_fft,
            win_length=self.window_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0, # Nyqust range
            f_max=sample_rate / 2.0,
            power=2.0 # 1 for magnitude spectrum, 2 for power spectrum
        )
    
    def extract(self, audio):
        """
        Args:
            audio: Input audio as np.int16 array, output of PDM to PCM
            
        Returns:
            log_mel_spec: Log-mel spectrogram (n_mels x n_frames)
            time_frames: Time in seconds for each frame
        """
        # normalize to [-1, 1]
        audio_tensor = torch.from_numpy(audio.astype(np.float32)) / 32768.0
        
        # compute mel spectrogram, shape is matrix of [n_mels, n_frames]
        # each column is one time frame
        # each row is one mel frequency band
        mel_spec = self.mel_transform(audio_tensor)
        
        # convert to log scale (dB)
        # human colume perception is logarithmic
        # change 10 to 20 if magnitude spectrum wanted
        log_mel_spec = 10 * torch.log10(mel_spec + 1e-10)
        
        # Time axis, each frame is 0.01 s
        time_frames = torch.arange(mel_spec.shape[1]) * self.hop_length / self.sample_rate
        
        # return feature matric and time base
        return log_mel_spec.numpy(), time_frames.numpy()
    
    def get_config(self):
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'hop_length': self.hop_length,
            'window_length': self.window_length
        }