import torch
import numpy as np

# PDM bitstream to decimator into PCM signal
class PDMMic:
    """
    Emulate PDM  output, 1.5 MHz to 3 MHz 
    for infineon mic i think 2.4 MHz is middle operating mode
    """
    def __init__(self, pdm_clock_rate=2400000, order=1):
        """
        Args:
            pdm_clock_rate: PDM clock frequency (typically 1-3 MHz)
            order: PDM modulator order (1 or 2)
        """
        self.pdm_clock_rate = pdm_clock_rate
        self.order = order
        self.state = 0.0  # Integrator state for delta-sigma modulation
    
    def analog_to_pdm(self, audio_pcm, pcm_sample_rate):
        """
        Convert PCM audio to PDM bitstream with sigma-delta modulation
        Args:
            audio_pcm: Input audio as normalized float [-1, 1]
            pcm_sample_rate: Sample rate of input PCM
            
        Returns:
            pdm_bitstream: 1-bit PDM output as uint8
        """
        # upsample PCM to PDM clock rate
        upsample_factor = self.pdm_clock_rate // pcm_sample_rate
        audio_upsampled = torch.repeat_interleave(
            torch.from_numpy(audio_pcm.astype(np.float32)), 
            upsample_factor
        )
        
        # First-order delta-sigma modulation
        pdm_bits = torch.zeros(len(audio_upsampled), dtype=torch.uint8)
        integrator = 0.0
        
        for i in range(len(audio_upsampled)):
            # add input to integrator
            integrator += audio_upsampled[i].item()
            
            # Quantize: output PDM bit 1 if integrator >= 0 and decrement, else 0 and increment
            if integrator >= 0:
                pdm_bits[i] = 1
                integrator -= 1.0  # Feedback
            else:
                pdm_bits[i] = 0
                integrator += 1.0  # Feedback
        
        # tensor to ndarray
        return pdm_bits.numpy()
    
    def get_config(self):
        return {
            'pdm_clock_rate': self.pdm_clock_rate,
            'order': self.order
        }

# Cascaded Integrator-Comb
class CICDecimator:
    def __init__(self, decimation_factor, num_stages=3, differential_delay=1):
        """
        Args:
            R - decimation_factor: Decimation ratio 
            N - num_stages: # of CIC stages, normally 3 to 5
            M - differential_delay: Differential delay, usuallt 1 or 2
        """
        self.decimation_factor = decimation_factor
        self.num_stages = num_stages
        self.differential_delay = differential_delay
        
        # accumulators running at PDM clock rate
        self.integrator_states = [0] * num_stages
        
        # delay lines for comb stages, running at decimated rate
        self.comb_delays = [[0] * differential_delay for i in range(num_stages)]
    
    def process(self, pdm_bitstream):
        """
        Decimate PDM bitstream to PCM
        
        Args:
            pdm_bitstream: 1-bit PDM input
        Returns:
            pcm_output: Decimated PCM output as int32
        """

        # make tensor from a np array
        pdm_bits = torch.from_numpy(pdm_bitstream.astype(np.int32))
        
        # calc output length and make zero array of that size
        output_len = len(pdm_bits) // self.decimation_factor
        pcm_output = torch.zeros(output_len, dtype=torch.int32)


        output_idx = 0
        # loop for every PDM bit we'll have
        for i in range(len(pdm_bits)):
            sample = pdm_bits[i].item()
            
            # integrator stages at PDM rate
            # H(z) = 1 / (1 - z^(-1))^N
            # accumulate PDM density
            # also LPF
            for stage in range(self.num_stages):
                self.integrator_states[stage] += sample
                sample = self.integrator_states[stage]
            
            # Decimation: only process every Rth sample
            if (i + 1) % self.decimation_factor == 0:
                # comb stages at decimated rate
                for stage in range(self.num_stages):
                    # Differentiation: y[n] = x[n] - x[n-D]
                    # H(z) = (1 - z^(-M))^N
                    # also HPF
                    delayed = self.comb_delays[stage][0]
                    diff = sample - delayed
                    
                    # update delay line
                    for d in range(len(self.comb_delays[stage]) - 1):
                        self.comb_delays[stage][d] = self.comb_delays[stage][d + 1]
                    self.comb_delays[stage][-1] = sample
                    
                    sample = diff
                
                # Write raw PCM data, need to normalize
                if output_idx < output_len:
                    pcm_output[output_idx] = sample
                    output_idx += 1
        
        return pcm_output.numpy()
    
    def reset(self):
        self.integrator_states = [0] * self.num_stages
        self.comb_delays = [[0] * self.differential_delay for i in range(self.num_stages)]
    
    # calculate filter DC gain for normalization, divide by this later
    def get_gain(self):
        return (self.decimation_factor * self.differential_delay) ** self.num_stages
    
    def get_config(self):
        return {
            'decimation_factor': self.decimation_factor,
            'num_stages': self.num_stages,
            'differential_delay': self.differential_delay,
            'gain': self.get_gain()
        }


class PDMToPCMConverter:
    """
    Complete PDM to PCM by combining PDM mic emulation and CIC decimation
    """
    def __init__(self, pdm_clock_rate=2400000, output_sample_rate=16000, 
                 cic_stages=3):
        """
        Args:
            pdm_clock_rate: PDM clock frequency
            output_sample_rate: Desired PCM output rate
            cic_stages: # of CIC filter stages
        """
        self.pdm_clock_rate = pdm_clock_rate
        self.output_sample_rate = output_sample_rate
        self.decimation_factor = pdm_clock_rate // output_sample_rate
        
        self.pdm_mic = PDMMic(pdm_clock_rate=pdm_clock_rate)
        self.cic = CICDecimator(
            decimation_factor=self.decimation_factor,
            num_stages=cic_stages
        )
    
    def process(self, audio_pcm, input_sample_rate):
        """
        Actually process and convert PDM to normalized PCM
        Args:
            audio_pcm: Input audio normalized to [-1, 1]
            input_sample_rate: Sample rate of input
            
        Returns:
            pcm_output: Decimated PCM as int16
        """
        # Convert input audio into PDM bitstream
        pdm_bits = self.pdm_mic.analog_to_pdm(audio_pcm, input_sample_rate)
        
        # Decimate it
        pcm_raw = self.cic.process(pdm_bits)
        
        # Normalize by CIC gain factor and then scale to int
        gain = self.cic.get_gain()
        pcm_normalized = pcm_raw.astype(np.float32) / gain
        
        # Scale to int16 range
        pcm_int16 = np.clip(pcm_normalized * 32768, -32768, 32767).astype(np.int16)
        
        return pcm_int16
    
    def reset(self):
        self.cic.reset()
    
    def get_config(self):
        return {
            'pdm_mic': self.pdm_mic.get_config(),
            'cic': self.cic.get_config(),
            'decimation_factor': self.decimation_factor,
            'output_sample_rate': self.output_sample_rate
        }