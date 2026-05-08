import numpy as np
from golden_model_spectral_vad import (
    FullPipelineGoldenExtractor, VadSpectral,
    _pcm_to_cic_input, _CICDecimator, _CompFIR,
    GoldenExtractor, N_FFT, HOP_LENGTH, N_BINS,
    SAMPLE_RATE, SAMPLE_MAX, POWER_MASK
)

def measure_frame_energies(pcm):
    """Run through pipeline, return per-frame spectral energy sums."""
    cic = _CICDecimator()
    fir = _CompFIR()
    stfft = GoldenExtractor()

    pcm = np.clip(np.asarray(pcm, dtype=np.int32), -32768, 32767)
    cic_in = _pcm_to_cic_input(pcm, decim=63)
    cic_raw = cic.process(cic_in)
    cic_trunc = _CICDecimator.truncate(cic_raw)
    fir_raw = fir.process(cic_trunc)
    fir_trunc = _CompFIR.truncate(fir_raw)

    n = len(fir_trunc)
    n_frames = max(0, (n - N_FFT) // HOP_LENGTH + 1)
    energies = []

    for f in range(n_frames):
        s = f * HOP_LENGTH
        frame = fir_trunc[s : s + N_FFT].astype(np.int32)
        windowed = stfft._window_frame(frame)
        re, im, _ = stfft._fft_frame(windowed)
        power = stfft._power(re, im)
        energy = min(int(np.sum(power.astype(np.uint64))), (1 << 32) - 1)
        energies.append(energy)

    return np.array(energies, dtype=np.uint64)


# Generate test signals
t = np.arange(7500) / SAMPLE_RATE
silence = np.zeros(7500, dtype=np.int32)
chirp = (np.sin(2 * np.pi * (200 * t + (7000-200)/(2*0.47) * t**2)) * SAMPLE_MAX).astype(np.int32)

silence_energies = measure_frame_energies(silence)
chirp_energies = measure_frame_energies(chirp)

print("=== Silence frame energies ===")
print(f"  Count: {len(silence_energies)}")
print(f"  Min:   {silence_energies.min()}")
print(f"  Max:   {silence_energies.max()}")
print(f"  Mean:  {silence_energies.mean():.0f}")

print()
print("=== Chirp frame energies ===")
print(f"  Count: {len(chirp_energies)}")
print(f"  Min:   {chirp_energies.min()}")
print(f"  Max:   {chirp_energies.max()}")
print(f"  Mean:  {chirp_energies.mean():.0f}")

from pathlib import Path

wav_dir = Path("/workspace/src/rtl/top/speech_data")
if wav_dir.exists():
    from scipy.io import wavfile
    all_speech_energies = []
    for kw_dir in sorted(wav_dir.iterdir()):
        if not kw_dir.is_dir():
            continue
        for wav_path in sorted(kw_dir.glob("*.wav"))[:2]:
            rate, data = wavfile.read(str(wav_path))
            pcm_f = data.astype(np.float64) / 32768.0
            target = int(0.47 * SAMPLE_RATE)
            if len(pcm_f) >= target:
                pcm_f = pcm_f[:target]
            else:
                pcm_f = np.concatenate([pcm_f, np.zeros(target - len(pcm_f))])
            pcm = np.clip(np.round(pcm_f * 32767), -32768, 32767).astype(np.int32)
            energies = measure_frame_energies(pcm)
            all_speech_energies.extend(energies.tolist())
            print(f"  {kw_dir.name}/{wav_path.name}: min={energies.min()} max={energies.max()} mean={energies.mean():.0f}")

    all_speech = np.array(all_speech_energies, dtype=np.uint64)
    print(f"\n=== All speech frame energies ===")
    print(f"  Count: {len(all_speech)}")
    print(f"  Min:   {all_speech.min()}")
    print(f"  Max:   {all_speech.max()}")
    print(f"  Mean:  {all_speech.mean():.0f}")

print()
# Find the gap
silence_max = int(silence_energies.max())
chirp_min = int(chirp_energies.min())
suggested = (silence_max + chirp_min) // 2
print(f"Silence max: {silence_max}")
print(f"Chirp min:   {chirp_min}")
print(f"Suggested threshold: {suggested}")
print(f"  (midpoint between silence ceiling and speech floor)")