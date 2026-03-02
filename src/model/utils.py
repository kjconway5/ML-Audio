# WAVE Files, I/O, and Testing Utility Functions
import wave
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import signal
from pathlib import Path

from features import LogMelExtractor


# WAV I/O
def load_wav(filepath):
    with wave.open(str(filepath), 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        audio_data = wav.readframes(wav.getnframes())
        audio = np.frombuffer(audio_data, dtype=np.int16)
        
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    return audio, sample_rate


def save_wav(filepath, audio, sample_rate):
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


def visualize_pipeline(original, filtered_low, filtered_high, features_low, features_high,
                sample_rate, config_low, config_high):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 16),
                             constrained_layout=True)
    
    fig.suptitle('Preprocessing and Spectrograms', fontsize=18, fontweight='bold')
    
    duration = len(original) / sample_rate
    time = np.linspace(0, duration, len(original))
    
    waveform_max = max(np.max(np.abs(original)),
                       np.max(np.abs(filtered_low)),
                       np.max(np.abs(filtered_high)))
    waveform_ylim = [-waveform_max*1.1, waveform_max*1.1]
    
    # top row waveforms
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
    
    # middle row STFT Spectrograms
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
    
    # bottom row Log-mel features
    extractor = LogMelExtractor(sample_rate=sample_rate)
    features_orig, t_orig = extractor.extract(original)
    _, t_low = extractor.extract(filtered_low)
    _, t_high = extractor.extract(filtered_high)
    
    logmel_data = [(features_orig, t_orig), (features_low, t_low), (features_high, t_high)]
    logmel_titles = [
        'Log-Mel Features (Original)',
        'Log-Mel Features (Low-Order)',
        'Log-Mel Features (High-Order)'
    ]
    
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


# RTL test vectors
def generate_rtl_test_vectors(audio, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    hex_file = output_dir / f'{prefix}_hex.txt'
    with open(hex_file, 'w') as f:
        for sample in audio:
            unsigned = int(sample) & 0xFFFF
            f.write(f'{unsigned:04X}\n')
    
    bin_file = output_dir / f'{prefix}_bin.txt'
    with open(bin_file, 'w') as f:
        for sample in audio:
            unsigned = int(sample) & 0xFFFF
            f.write(f'{unsigned:016b}\n')
    
    print(f"Generated RTL vectors: {prefix}_hex.txt, {prefix}_bin.txt")