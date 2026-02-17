import torch
import torchaudio.functional as F
import numpy as np

# Configurable cascaded biquads built-in with pytorch
class CascadedBiquadFilter:
    def __init__(self, sample_rate=16000, hpf_order=2, lpf_order=4,
                 cutoff_hpf=150, cutoff_lpf=4000):
        """
        Args:
            sample_rate: in Hz
            hpf_order: HPF order must be even
            lpf_order: LPF order must be even
            cutoff_hpf: cutoff frequency
            cutoff_lpf: cutoff frequency
        """
        self.sample_rate = sample_rate
        self.hpf_order = hpf_order
        self.lpf_order = lpf_order
        self.cutoff_hpf = cutoff_hpf
        self.cutoff_lpf = cutoff_lpf
        
        # at least 1, floor division if arg % 2 != 0
        self.hpf_stages = max(1, hpf_order // 2)
        self.lpf_stages = max(1, lpf_order // 2)
    
    def process(self, audio):
        """
        Args:
            audio: Input audio as np.int16 array
            
        Returns:
            Filtered audio as np.int16 array
        """
        # normalize int16 -> float [-1,1]
        audio_float = torch.from_numpy(audio.astype(np.float32)) / 32768.0
        
        audio_filtered = audio_float.clone()
        
        # HPF stages
        for i in range(self.hpf_stages):
            audio_filtered = F.highpass_biquad(
                audio_filtered, 
                self.sample_rate, 
                self.cutoff_hpf, 
                Q=0.707
            )
        
        # LPF stages
        for i in range(self.lpf_stages):
            audio_filtered = F.lowpass_biquad(
                audio_filtered, 
                self.sample_rate, 
                self.cutoff_lpf, 
                Q=0.707
            )
        
        # scale back to int16
        audio_filtered = torch.clamp(audio_filtered * 32768.0, -32768, 32767)
        return audio_filtered.numpy().astype(np.int16)
    
    def get_config(self):
        return {
            'sample_rate': self.sample_rate,
            'hpf_order': self.hpf_order,
            'lpf_order': self.lpf_order,
            'cutoff_hpf': self.cutoff_hpf,
            'cutoff_lpf': self.cutoff_lpf,
            'total_biquads': self.hpf_stages + self.lpf_stages
        }