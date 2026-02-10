import numpy as np
from pdm_decimator import PDMToPCMConverter
from fifo import FIFO
from filters import CascadedBiquadFilter
from features import LogMelExtractor



class Preprocessing:
    def __init__(self, sample_rate=16000, hpf_order=2, lpf_order=4, cutoff_hpf=150,
                 cutoff_lpf=4000, n_mels=64, n_fft=512, hop_length=160):
        """
        Args:
            sample_rate: Audio sample rate
            hpf_order, lpf_order: Filter orders
            cutoff_hpf, cutoff_lpf: Filter cutoff frequencies
            n_mels: Number of mel bins for ML model
            n_fft: FFT size
            hop_length: Hop between frames
        """
        self.filter = CascadedBiquadFilter(
            sample_rate=sample_rate,
            hpf_order=hpf_order,
            lpf_order=lpf_order,
            cutoff_hpf=cutoff_hpf,
            cutoff_lpf=cutoff_lpf
        )
        
        self.feature_extractor = LogMelExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    def process(self, audio):
        """
        Args:
            audio: Input audio as np.int16 array
            
        Returns:
            features: Log-mel features (n_mels x n_frames)
            filtered_audio
            time_frames: Time axis for features
        """
        # filter stage
        filtered_audio = self.filter.process(audio)
        
        # feature extraction
        features, time_frames = self.feature_extractor.extract(filtered_audio)
        
        return features, filtered_audio, time_frames
    
    def get_config(self):
        return {
            'filter': self.filter.get_config(),
            'features': self.feature_extractor.get_config()
        }


class Pipeline:
    """
    PDM mic, CIC, FIFO, Filters, then log-mel
    """
    
    def __init__(self, pdm_clock_rate=2400000,
                 output_sample_rate=16000,
                 fifo_depth=2048,
                 chunk_size=512,
                 hpf_order=2,
                 lpf_order=4,
                 n_mels=64):
        # PDM and CIC
        self.pdm_converter = PDMToPCMConverter(
            pdm_clock_rate=pdm_clock_rate,
            output_sample_rate=output_sample_rate
        )
        
        # FIFO buffer
        self.fifo = FIFO(
            depth=fifo_depth,
            chunk_size=chunk_size
        )
        
        # filters and log-mel features
        self.preprocessor = Preprocessing(
            sample_rate=output_sample_rate,
            hpf_order=hpf_order,
            lpf_order=lpf_order,
            n_mels=n_mels
        )
    
    def process(self, audio_input, input_sample_rate):
        """
        Args:
            audio_input: Input audio (normalized float or int16)
            input_sample_rate: Input sample rate
            
        Returns:
            features: Log-mel features for ML model
            intermediate: Dict with intermediate outputs for debugging
        """
        # normalize if int16
        if audio_input.dtype == np.int16:
            audio_normalized = audio_input.astype(np.float32) / 32768.0
        else:
            audio_normalized = audio_input
        
        # CIC decimation of PDM input
        pcm_decimated = self.pdm_converter.process(audio_normalized, input_sample_rate)
        
        # write data, read it and process it
        self.fifo.write(pcm_decimated)
        all_features = []
        
        while self.fifo.get_chunks_available() > 0:
            chunk = self.fifo.read_chunk()
            features, filtered, _ = self.preprocessor.process(chunk)
            all_features.append(features)
        
        if all_features:
            combined_features = np.concatenate(all_features, axis=1)
        else:
            combined_features = None
        
        intermediate = {
            'pcm_decimated': pcm_decimated,
            'fifo_count': self.fifo.get_count()
        }
        
        return combined_features, intermediate
    
    def get_config(self):
        return {
            'pdm_converter': self.pdm_converter.get_config(),
            'fifo': self.fifo.get_config(),
            'preprocessor': self.preprocessor.get_config()
        }
    

# simplified pipeline to take in WAV files, skipping PDM/CIC
# and optionally applying filters, then straight to log-mel
class SimplePipeline:
    def __init__(self, sample_rate=16000,
                 use_filters=True,
                 hpf_order=2,
                 lpf_order=4,
                 cutoff_hpf=150,
                 cutoff_lpf=4000,
                 n_mels=64,
                 n_fft=512,
                 hop_length=160,
                 window_length=None):
        """
        Args:
            sample_rate: Input rate, 16 kHz for Speech Commands
            use_filters: If False, skip filtering 
            hpf_order: must be even
            lpf_order: must be even
            cutoff_hpf: HPF cutoff frequency in Hz
            cutoff_lpf: LPF cutoff frequency in Hz
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop between frames
        """
        self.sample_rate = sample_rate
        self.use_filters = use_filters
        
        # create filter only if enabled
        if use_filters:
            self.filter = CascadedBiquadFilter(
                sample_rate=sample_rate,
                hpf_order=hpf_order,
                lpf_order=lpf_order,
                cutoff_hpf=cutoff_hpf,
                cutoff_lpf=cutoff_lpf
            )
        else:
            self.filter = None
        
        # features
        self.feature_extractor = LogMelExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            window_length=window_length or n_fft
        )
    
    def process(self, audio_input):
        """
        Args:
            audio_input: Complete audio as np.int16 array (from load_wav)
            
        Returns:
            features: Log-mel features (n_mels x n_frames)
            filtered_audio: Filtered audio (or original if no filters)
        """
        # apply filters if enabled
        if self.use_filters:
            filtered_audio = self.filter.process(audio_input)
        else:
            filtered_audio = audio_input
        
        # extract
        features, _ = self.feature_extractor.extract(filtered_audio)
        
        return features, filtered_audio
    
    def get_config(self):
        config = {
            'sample_rate': self.sample_rate,
            'use_filters': self.use_filters,
            'feature_extractor': self.feature_extractor.get_config()
        }
        if self.use_filters:
            config['filter'] = self.filter.get_config()
        return config
