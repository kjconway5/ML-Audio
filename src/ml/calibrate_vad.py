import numpy as np
from golden_model_spectral_vad import (
    FullPipelineGoldenExtractor, VadSpectral,
    _pcm_to_cic_input, _CICDecimator, _CompFIR,
    GoldenExtractor, N_FFT, HOP_LENGTH, N_BINS,
    SAMPLE_RATE, SAMPLE_MAX, POWER_MASK
)

CALIB_FRAMES = 256


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


# =========================================================================
# Part 1: Baseline — pure silence and full chirp
# =========================================================================
print("=" * 70)
print("PART 1: Baseline signals")
print("=" * 70)

t = np.arange(7500) / SAMPLE_RATE
silence = np.zeros(7500, dtype=np.int32)
chirp = (np.sin(2 * np.pi * (200 * t + (7000-200)/(2*0.47) * t**2)) * SAMPLE_MAX).astype(np.int32)

silence_energies = measure_frame_energies(silence)
chirp_energies = measure_frame_energies(chirp)

# Skip frame 0 (startup transient) for steady-state analysis
silence_steady = silence_energies[1:]
silence_e0 = int(silence_energies[0])
silence_steady_val = int(silence_steady[0]) if len(silence_steady) > 0 else 0

print(f"\n  Silence:")
print(f"    Frame 0 (startup): {silence_e0:>12,}")
print(f"    Steady state:      {silence_steady_val:>12,}  (all {len(silence_steady)} frames identical)")
print(f"    Ratio spike/steady: {silence_e0/silence_steady_val:.1f}x")

print(f"\n  Chirp (full amplitude):")
print(f"    Min: {chirp_energies.min():>12,}")
print(f"    Max: {chirp_energies.max():>12,}")
print(f"    Mean: {chirp_energies.mean():>12,.0f}")


# =========================================================================
# Part 2: Low-amplitude signals — find the boundary
# =========================================================================
print()
print("=" * 70)
print("PART 2: Low-amplitude signals (where does energy rise above silence?)")
print("=" * 70)

# Generate chirps at various amplitudes and measure frame energies
# We want to find at what amplitude the signal energy starts to
# meaningfully exceed the silence floor

amplitudes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
              0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

print(f"\n  {'Amplitude':>10s}  {'PCM peak':>10s}  {'Min E':>12s}  {'Mean E':>12s}  {'Max E':>12s}  "
      f"{'vs silence':>10s}  {'>5.2M':>6s}  {'>8.4M':>6s}  {'>10.5M':>6s}  {'>11M':>6s}")
print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  "
      f"{'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

threshold_candidates = {
    '5.2M': 5_200_000,    # 1.25x mean
    '8.4M': 8_400_000,    # 2x mean
    '10.5M': 10_500_000,  # 2.5x mean
    '11M': 11_000_000,    # hand-tuned fixed
}

for amp in amplitudes:
    pcm_amp = (np.sin(2 * np.pi * (200 * t + (7000-200)/(2*0.47) * t**2)) * SAMPLE_MAX * amp).astype(np.int32)
    e = measure_frame_energies(pcm_amp)
    e_skip1 = e[1:]  # skip startup frame

    above_5_2  = int(np.sum(e_skip1 > 5_200_000))
    above_8_4  = int(np.sum(e_skip1 > 8_400_000))
    above_10_5 = int(np.sum(e_skip1 > 10_500_000))
    above_11   = int(np.sum(e_skip1 > 11_000_000))
    total = len(e_skip1)

    pcm_peak = int(abs(pcm_amp).max())
    ratio_vs_silence = float(e_skip1.mean()) / silence_steady_val if silence_steady_val > 0 else 0

    print(f"  {amp:>10.3f}  {pcm_peak:>10,}  {e_skip1.min():>12,}  {e_skip1.mean():>12,.0f}  {e_skip1.max():>12,}  "
          f"{ratio_vs_silence:>9.2f}x  "
          f"{above_5_2:>2d}/{total}  {above_8_4:>2d}/{total}  {above_10_5:>2d}/{total}  {above_11:>2d}/{total}")


# =========================================================================
# Part 3: White noise at various amplitudes
# =========================================================================
print()
print("=" * 70)
print("PART 3: White noise (random, more realistic than chirp)")
print("=" * 70)

np.random.seed(42)

print(f"\n  {'Amplitude':>10s}  {'RMS':>8s}  {'Min E':>12s}  {'Mean E':>12s}  {'Max E':>12s}  "
      f"{'vs silence':>10s}  {'>5.2M':>6s}  {'>10.5M':>6s}  {'>11M':>6s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  "
      f"{'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}")

noise_amps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
              0.07, 0.10, 0.15, 0.20, 0.30, 0.50]

for amp in noise_amps:
    noise = (np.random.randn(7500) * SAMPLE_MAX * amp).astype(np.int32)
    noise = np.clip(noise, -32768, 32767)
    e = measure_frame_energies(noise)
    e_skip1 = e[1:]

    rms = float(np.sqrt(np.mean(noise.astype(np.float64)**2)))
    above_5_2  = int(np.sum(e_skip1 > 5_200_000))
    above_10_5 = int(np.sum(e_skip1 > 10_500_000))
    above_11   = int(np.sum(e_skip1 > 11_000_000))
    total = len(e_skip1)
    ratio_vs_silence = float(e_skip1.mean()) / silence_steady_val if silence_steady_val > 0 else 0

    print(f"  {amp:>10.3f}  {rms:>8.0f}  {e_skip1.min():>12,}  {e_skip1.mean():>12,.0f}  {e_skip1.max():>12,}  "
          f"{ratio_vs_silence:>9.2f}x  "
          f"{above_5_2:>2d}/{total}  {above_10_5:>2d}/{total}  {above_11:>2d}/{total}")


# =========================================================================
# Part 4: Single tone at various amplitudes (narrowband)
# =========================================================================
print()
print("=" * 70)
print("PART 4: 1kHz tone at various amplitudes")
print("=" * 70)

print(f"\n  {'Amplitude':>10s}  {'Min E':>12s}  {'Mean E':>12s}  {'Max E':>12s}  "
      f"{'vs silence':>10s}  {'>5.2M':>6s}  {'>10.5M':>6s}  {'>11M':>6s}")
print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  "
      f"{'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}")

tone_amps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
             0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

for amp in tone_amps:
    tone = (np.sin(2 * np.pi * 1000 * t) * SAMPLE_MAX * amp).astype(np.int32)
    e = measure_frame_energies(tone)
    e_skip1 = e[1:]

    above_5_2  = int(np.sum(e_skip1 > 5_200_000))
    above_10_5 = int(np.sum(e_skip1 > 10_500_000))
    above_11   = int(np.sum(e_skip1 > 11_000_000))
    total = len(e_skip1)
    ratio_vs_silence = float(e_skip1.mean()) / silence_steady_val if silence_steady_val > 0 else 0

    print(f"  {amp:>10.3f}  {e_skip1.min():>12,}  {e_skip1.mean():>12,.0f}  {e_skip1.max():>12,}  "
          f"{ratio_vs_silence:>9.2f}x  "
          f"{above_5_2:>2d}/{total}  {above_10_5:>2d}/{total}  {above_11:>2d}/{total}")


# =========================================================================
# Part 5: Auto-calibration with warmup skip simulation
# =========================================================================
print()
print("=" * 70)
print("PART 5: Auto-calibration with warmup skip")
print("=" * 70)

silence_long = np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.int32)
silence_long_energies = measure_frame_energies(silence_long)

for warmup in [0, 1, 2, 4, 8]:
    post_warmup = silence_long_energies[warmup:]
    calib_window = post_warmup[:CALIB_FRAMES]
    post_calib = post_warmup[CALIB_FRAMES:]

    calib_sum = int(np.sum(calib_window.astype(np.uint64)))
    calib_mean = calib_sum >> 8  # CALIB_FRAMES=256 -> shift by 8
    calib_max = int(calib_window.max())

    # Test various multipliers with this warmup
    print(f"\n  Warmup={warmup} frames skipped:")
    print(f"    Calib window: frames {warmup} to {warmup+CALIB_FRAMES-1}")
    print(f"    Calib mean: {calib_mean:>12,}   Calib max: {calib_max:>12,}")

    for mult_name, thresh in [
        ("1.25x mean", calib_mean + (calib_mean >> 2)),
        ("2x mean",    calib_mean * 2),
        ("2.5x mean",  (calib_mean << 1) + (calib_mean >> 1)),
        ("3x mean",    calib_mean * 3),
        ("1.25x max",  calib_max + (calib_max >> 2)),
    ]:
        chirp_above = int(np.sum(chirp_energies > thresh))
        silence_leak = int(np.sum(post_calib > thresh))
        print(f"    {mult_name:<12s} = {thresh:>12,}  vs11M={thresh/11_000_000:>5.2f}x  "
              f"chirp={chirp_above}/{len(chirp_energies)}  "
              f"silence_leak={silence_leak}/{len(post_calib)}")


# =========================================================================
# Part 6: Speech retention at key thresholds
# =========================================================================
from pathlib import Path
wav_dir = Path("/workspace/src/rtl/top/speech_data")
if wav_dir.exists():
    from scipy.io import wavfile

    print()
    print("=" * 70)
    print("PART 6: Speech retention comparison")
    print("=" * 70)

    all_speech_energies = []
    for kw_dir in sorted(wav_dir.iterdir()):
        if not kw_dir.is_dir():
            continue
        for wav_path in sorted(kw_dir.glob("*.wav"))[:20]:
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

    all_speech = np.array(all_speech_energies, dtype=np.uint64)
    speech_total = len(all_speech)
    print(f"\n  Speech dataset: {speech_total} frames from 20 files/keyword")
    print(f"  Min: {all_speech.min():,}  Max: {all_speech.max():,}  Mean: {all_speech.mean():,.0f}")

    # Use warmup=1 calibration mean (skip just frame 0)
    calib_after_warmup = silence_long_energies[1:1+CALIB_FRAMES]
    cal_mean = int(np.sum(calib_after_warmup.astype(np.uint64))) >> 8

    post_cal = silence_long_energies[1+CALIB_FRAMES:]

    print(f"\n  Calibration mean (warmup=1): {cal_mean:,}")
    print(f"\n  {'Method':<20s}  {'Threshold':>12s}  {'vs 11M':>8s}  {'Speech kept':>16s}  {'Silence leak':>14s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*16}  {'-'*14}")

    methods = [
        ("Fixed 11M",           11_000_000),
        ("Fixed 10M",           10_000_000),
        ("1.25x mean",          cal_mean + (cal_mean >> 2)),
        ("1.5x mean",           cal_mean + (cal_mean >> 1)),
        ("2x mean",             cal_mean * 2),
        ("2.5x mean",           (cal_mean << 1) + (cal_mean >> 1)),
        ("3x mean",             cal_mean * 3),
    ]

    for name, thresh in methods:
        speech_above = int(np.sum(all_speech > thresh))
        silence_leak = int(np.sum(post_cal > thresh))
        print(f"  {name:<20s}  {thresh:>12,}  {thresh/11_000_000:>7.2f}x  "
              f"{speech_above:>7,}/{speech_total} ({100.0*speech_above/speech_total:>5.1f}%)  "
              f"{silence_leak:>4d}/{len(post_cal)} ({100.0*silence_leak/len(post_cal):>5.1f}%)")