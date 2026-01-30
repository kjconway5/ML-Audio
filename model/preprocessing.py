# Audio Preprocessing
# Configurable Cascaded Biquads (HPF and LPF), visualization, and Log-Mel Feature Extraction for model
import wave
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy import signal
from pathlib import Path


class CascadedBiquadFilter:
    """
    Implement cascaded biquads, each biquad is a 2nd order filter
    """
    def __init__(self, sample_rate=16000, hpf_order=2, lpf_order=4,
                 cutoff_hpf=150, cutoff_lpf=4000):
        """
        Initialize cascaded biquad filter
        Args:
            sample_rate in Hz
            hpf_order (needs to be even)
            lpf_order (needs to be even)
            cutoff_hpf
            cutoff_lpf
        """
        self.sample_rate = sample_rate
        self.hpf_order = hpf_order
        self.lpf_order = lpf_order
        self.cutoff_hpf = cutoff_hpf
        self.cutoff_lpf = cutoff_lpf
        
        self.hpf_stages = max(1, hpf_order // 2)
        self.lpf_stages = max(1, lpf_order // 2)
    
    def process(self, audio):
        """
        Args:
            audio: Input audio as np.int16 array
        Returns:
            Filtered audio as np.int16 array
        """
        # normalize int16 to float [-1,1]
        audio_float = torch.from_numpy(audio.astype(np.float32)) / 32768.0
        
        audio_filtered = audio_float.clone()
        
        # apply HPF
        for i in range(self.hpf_stages):
            audio_filtered = F.highpass_biquad(
                audio_filtered, 
                self.sample_rate, 
                self.cutoff_hpf, 
                Q=0.707
            )
        
        # apply LPF
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


class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=64, 
                 hop_length=160, window_length=None):
        """
        Args:
            sample_rate in Hz
            n_fft - FFT size
            n_mels - Number of mel bins
            hop_length - Num samples b/t successive frames
            window_length - (defaults to n_fft)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.window_length = window_length or n_fft
        
        # create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.window_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            power=2.0
        )
    
    def extract(self, audio):
        """
        Args:
            audio - Input audio as np.int16 array
        Returns:
            log_mel_spec - Log-mel spectrogram (n_mels x n_frames)
            time_frames - Time in seconds for each frame
        """
        # normalize to [-1, 1]
        audio_tensor = torch.from_numpy(audio.astype(np.float32)) / 32768.0
        
        # compute mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # convert to log scale (dB)
        log_mel_spec = 10 * torch.log10(mel_spec + 1e-10)
        # time axis
        time_frames = torch.arange(mel_spec.shape[1]) * self.hop_length / self.sample_rate
        
        return log_mel_spec.numpy(), time_frames.numpy()
    
    def get_config(self):
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'hop_length': self.hop_length,
            'window_length': self.window_length
        }


class AudioPreprocessor:
    def __init__(self, 
                 sample_rate=16000,
                 hpf_order=2, 
                 lpf_order=4,
                 cutoff_hpf=150,
                 cutoff_lpf=4000,
                 n_mels=64,
                 n_fft=512,
                 hop_length=160):
        """
        Args:
            sample_rate in Hz
            hpf_order, lpf_order
            cutoff_hpf, cutoff_lpf - cutoff freqs
            n_mels - Number of mel bins for ML model
            n_fft - FFT size
            hop_length between frames
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
            audio - Input audio as np.int16 array
            
        Returns:
            features - Log-mel features for ML model (n_mels x n_frames)
            filtered_audio - Filtered audio
            time_frames - Time axis for features
        """
        print(f"Input: {len(audio)} samples")
        
        # filter stage
        filtered_audio = self.filter.process(audio)
        
        # feature extraction
        print(f"Extracting log-mel features ({self.feature_extractor.n_mels} mels)...")
        features, time_frames = self.feature_extractor.extract(filtered_audio)
        
        print(f"Output features: {features.shape}")
        print(f"Time frames: {len(time_frames)}")
        
        return features, filtered_audio, time_frames
    
    def get_config(self):
        return {
            'filter': self.filter.get_config(),
            'features': self.feature_extractor.get_config()
        }


# WAV File IO stuffs
def load_wav(filepath):
    """
    Returns:
        audio - Audio data as np.int16
        sample_rate in Hz
    """
    with wave.open(str(filepath), 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        audio_data = wav.readframes(wav.getnframes())
        audio = np.frombuffer(audio_data, dtype=np.int16)
        
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    return audio, sample_rate


def save_wav(filepath, audio, sample_rate):
    """
    Args:
        filepath - Output file path
        audio - Audio data as np.int16
        sample_rate in Hz
    """
    with wave.open(str(filepath), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())


# Test signal generators
def generate_chirp(duration=3, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration))
    chirp = signal.chirp(t, f0=50, f1=8000, t1=duration, method='linear')
    return np.int16(chirp * 32767 * 0.8)


def generate_multitone(duration=3, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequencies = [100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
    
    audio = np.zeros_like(t)
    for freq in frequencies:
        audio += np.sin(2 * np.pi * freq * t)
    
    # normalize
    audio = audio / np.max(np.abs(audio))
    return np.int16(audio * 32767 * 0.8)


# Visualization
def compute_spectrogram(audio, sample_rate, n_fft=512, hop_length=160):
    audio_tensor = torch.from_numpy(audio.astype(np.float32)) / 32768.0
    
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio_tensor, 
                      n_fft=n_fft,
                      hop_length=hop_length,
                      window=window,
                      return_complex=True)
    
    Sxx = torch.abs(stft) ** 2
    Sxx_db = 10 * torch.log10(Sxx + 1e-10)
    
    freqs = torch.linspace(0, sample_rate/2, n_fft//2 + 1)
    times = torch.arange(Sxx.shape[1]) * hop_length / sample_rate
    
    return freqs.numpy(), times.numpy(), Sxx_db.numpy()


def visualize_pipeline(original, filtered_low, filtered_high, 
                       features_low, features_high,
                       sample_rate,
                       config_low, config_high):
    """
    Args:
        original - Original audio
        filtered_low, filtered_high - Filtered audio outputs
        features_low, features_high - Log-mel features
        sample_rate
        config_low, config_high - Filter configurations
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 16),
                             constrained_layout=True)
    
    fig.suptitle('Preprocessing and Spectrograms', 
                 fontsize=18, fontweight='bold')
    
    duration = len(original) / sample_rate
    time = np.linspace(0, duration, len(original))
    
    # calculate shared axes
    waveform_max = max(np.max(np.abs(original)),
                       np.max(np.abs(filtered_low)),
                       np.max(np.abs(filtered_high)))
    waveform_ylim = [-waveform_max*1.1, waveform_max*1.1]
    
    # Top row for waveforms
    waveform_data = [original, filtered_low, filtered_high]
    titles_waveform = [
        'Original Waveform',
        f'Low-Order Filter\n({config_low["filter"]["total_biquads"]} biquads)',
        f'High-Order Filter\n({config_high["filter"]["total_biquads"]} biquads)'
    ]
    colors = ['blue', 'orange', 'red']
    
    for i, ax in enumerate(axes[0]):
        ax.plot(time, waveform_data[i], linewidth=0.5, color=colors[i], alpha=0.7)
        ax.set_title(titles_waveform[i], fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_ylim(waveform_ylim)
        ax.grid(True, alpha=0.3)
    
    # 2nd row for STFT Spectrograms
    stft_data = [original, filtered_low, filtered_high]
    stft_titles = [
        'STFT Spectrogram (Original)',
        'STFT Spectrogram (Low-Order)',
        'STFT Spectrogram (High-Order)'
    ]
    
    stfts = [compute_spectrogram(d, sample_rate) for d in stft_data]
    vmin = min(np.min(S) for _, _, S in stfts)
    vmax = max(np.max(S) for _, _, S in stfts)
    
    for i, ax in enumerate(axes[1]):
        f, t, S = stfts[i]
        im = ax.pcolormesh(t, f, S, shading='gouraud', cmap='viridis', 
                          vmin=vmin, vmax=vmax)
        ax.set_title(stft_titles[i], fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_ylim([0, 8000])
        ax.axhline(300, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(4000, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    # 3rd row for log-mel features
    logmel_specs = [
        features_low if filtered_low is not None else None,
        features_low,
        features_high
    ]
    logmel_titles = [
        'Log-Mel Features (Original)',
        'Log-Mel Features (Low-Order)',
        'Log-Mel Features (High-Order)'
    ]
    
    # calc original features for comparison
    extractor = LogMelExtractor(sample_rate=sample_rate)
    features_orig, t_orig = extractor.extract(original)
    _, t_low = extractor.extract(filtered_low)
    _, t_high = extractor.extract(filtered_high)
    
    logmel_data = [(features_orig, t_orig), (features_low, t_low), (features_high, t_high)]
    
    mel_vmin = min(np.min(feat) for feat, _ in logmel_data)
    mel_vmax = max(np.max(feat) for feat, _ in logmel_data)
    
    for i, ax in enumerate(axes[2]):
        feat, t_mel = logmel_data[i]
        im = ax.imshow(feat, aspect='auto', origin='lower', cmap='magma',
                       extent=[t_mel[0], t_mel[-1], 0, feat.shape[0]],
                       vmin=mel_vmin, vmax=mel_vmax)
        ax.set_title(logmel_titles[i], fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Mel Bin', fontsize=10)
        plt.colorbar(im, ax=ax, label='Log Power (dB)')
    
    return fig


# might help for RTL verif
def generate_rtl_test_vectors(audio, output_dir, prefix):
    """
    Args:
        audio - Audio data as np.int16
        output_dir - Output directory
        prefix - Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # hex format
    hex_file = output_dir / f'{prefix}_hex.txt'
    with open(hex_file, 'w') as f:
        for sample in audio:
            unsigned = int(sample) & 0xFFFF
            f.write(f'{unsigned:04X}\n')
    
    # binary format
    bin_file = output_dir / f'{prefix}_bin.txt'
    with open(bin_file, 'w') as f:
        for sample in audio:
            unsigned = int(sample) & 0xFFFF
            f.write(f'{unsigned:016b}\n')
    
    print(f"Generated RTL vectors: {prefix}_hex.txt, {prefix}_bin.txt")


# full processing function
def process_audio_file(input_wav, 
                       output_dir='output',
                       hpf_order_low=2,
                       lpf_order_low=4,
                       hpf_order_high=8,
                       lpf_order_high=8,
                       visualize=True,
                       save_outputs=True,
                       generate_test_vectors=True):
    """
    Args:
        input_wav - Path to input WAV file
        output_dir - Directory for outputs
        hpf_order_low, lpf_order_low - Low-order filter configuration
        hpf_order_high, lpf_order_high - High-order filter configuration
        visualize - Generate visualization plots
        save_outputs - Save filtered audio and features
        generate_test_vectors - Generate RTL test vectors
        
    Returns:
        results: Dictionary containing all outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # load audio
    print(f"\nLoading: {input_wav}")
    audio, sample_rate = load_wav(input_wav)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio)/sample_rate:.2f} seconds")
    print(f"Samples: {len(audio)}")
    
    # Create preprocessors
    print("\nLow-Order Configuration")
    preprocessor_low = AudioPreprocessor(
        sample_rate=sample_rate,
        hpf_order=hpf_order_low,
        lpf_order=lpf_order_low
    )
    print(f"Filter: {preprocessor_low.filter.get_config()}")
    
    print("\nHigh-Order Configuration")
    preprocessor_high = AudioPreprocessor(
        sample_rate=sample_rate,
        hpf_order=hpf_order_high,
        lpf_order=lpf_order_high
    )
    print(f"Filter: {preprocessor_high.filter.get_config()}")
    
    print("\nProcessing")
    print("Low-order pipeline:")
    features_low, filtered_low, time_low = preprocessor_low.process(audio)
    
    print("\nHigh-order pipeline:")
    features_high, filtered_high, time_high = preprocessor_high.process(audio)
    
    results = {
        'original_audio': audio,
        'filtered_low': filtered_low,
        'filtered_high': filtered_high,
        'features_low': features_low,
        'features_high': features_high,
        'time_frames_low': time_low,
        'time_frames_high': time_high,
        'sample_rate': sample_rate,
        'config_low': preprocessor_low.get_config(),
        'config_high': preprocessor_high.get_config()
    }
    
    # save outputs
    if save_outputs:
        print("\nSaving Outputs")
        save_wav(output_dir / 'filtered_low.wav', filtered_low, sample_rate)
        save_wav(output_dir / 'filtered_high.wav', filtered_high, sample_rate)
        
        np.save(output_dir / 'features_low.npy', features_low)
        np.save(output_dir / 'features_high.npy', features_high)
        print(f"  Saved filtered audio and features to {output_dir}/")
    
    # generate RTL test vectors
    if generate_test_vectors:
        print("\nGenerating RTL Test Vectors")
        generate_rtl_test_vectors(audio, output_dir, 'input_original')
        generate_rtl_test_vectors(filtered_low, output_dir, 'output_low')
        generate_rtl_test_vectors(filtered_high, output_dir, 'output_high')
    
    if visualize:
        print("\nGenerating Visualization")
        fig = visualize_pipeline(
            audio, filtered_low, filtered_high,
            features_low, features_high,
            sample_rate,
            preprocessor_low.get_config(),
            preprocessor_high.get_config()
        )
        fig.savefig(output_dir / 'pipeline_verification.png', dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {output_dir}/pipeline_verification.png")
        plt.show()

    
    return results


# demo
def main():
    print("\nChoose test signal:")
    print("1. Chirp")
    print("2. Multitone")
    print("3. Load existing WAV file")
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\nGenerating chirp test signal...")
        audio = generate_chirp(duration=3, sample_rate=16000)
        save_wav('chirp_test.wav', audio, 16000)
        input_wav = 'chirp_test.wav'
    elif choice == "2":
        print("\nGenerating multitone test signal...")
        audio = generate_multitone(duration=3, sample_rate=16000)
        save_wav('multitone_test.wav', audio, 16000)
        input_wav = 'multitone_test.wav'
    else:
        input_wav = input("Enter path to WAV file: ").strip()
    
    # process through pipeline
    results = process_audio_file(
        input_wav=input_wav,
        output_dir='output',
        hpf_order_low=2,
        lpf_order_low=4,
        hpf_order_high=8,
        lpf_order_high=8,
        visualize=True,
        save_outputs=True,
        generate_test_vectors=True
    )


if __name__ == "__main__":
    main()