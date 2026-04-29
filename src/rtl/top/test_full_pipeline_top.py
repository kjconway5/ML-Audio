
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

PCM_RATE = 16_000
DECIM    = 63
FFT_SIZE = 256
HOP      = FFT_SIZE // 2       # 128

N_PCM_SAMPLES = 7_500          # samples per synthesised test
N_MELS        = 40
Q_FRAC        = 10
DRAIN         = 100_000        # extra clocks after last PDM bit

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"

SPEECH_WAV_DIR  = Path(__file__).resolve().parent / "speech_data"
MAX_WAV_FILES   = 5            # cap to keep simulation time manageable
WAV_DURATION_S  = 0.47         # seconds per clip (= N_PCM_SAMPLES / PCM_RATE)


def _load_wav(path: Path) -> np.ndarray:
    """
    Load a .wav file and return int32 PCM at PCM_RATE=16000 Hz.

    Handles:
      - Mono and stereo (stereo averaged to mono)
      - Any sample rate (resampled to 16 kHz)
      - int16, int32, uint8, float32, float64 encodings
      - Trimming to WAV_DURATION_S or zero-padding if shorter
    """
    # Try scipy first, then soundfile
    data = None
    rate = None
    try:
        from scipy.io import wavfile as _wf
        rate, data = _wf.read(str(path))
    except Exception:
        pass

    if data is None:
        try:
            import soundfile as _sf
            data, rate = _sf.read(str(path))
        except Exception as e:
            raise RuntimeError(
                "Cannot load %s.\n"
                "Install scipy:    pip install scipy --break-system-packages\n"
                "or soundfile:     pip install soundfile --break-system-packages\n"
                "Original error:   %s" % (path, e)
            )

    # Normalise to float64 in [-1, 1]
    dt = np.asarray(data).dtype
    if dt == np.int16:
        pcm_f = data.astype(np.float64) / 32768.0
    elif dt == np.int32:
        pcm_f = data.astype(np.float64) / 2147483648.0
    elif dt == np.uint8:
        pcm_f = (data.astype(np.float64) - 128.0) / 128.0
    else:
        pcm_f = np.asarray(data, dtype=np.float64)

    # Stereo -> mono
    if pcm_f.ndim == 2:
        pcm_f = pcm_f.mean(axis=1)

    # Resample to PCM_RATE if needed
    if rate != PCM_RATE:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g     = gcd(int(PCM_RATE), int(rate))
            pcm_f = resample_poly(pcm_f, PCM_RATE // g, rate // g)
        except ImportError:
            # Fallback linear interpolation (lower quality)
            n_out = int(round(len(pcm_f) * PCM_RATE / rate))
            pcm_f = np.interp(
                np.linspace(0, 1, n_out),
                np.linspace(0, 1, len(pcm_f)),
                pcm_f,
            )

    # Trim or zero-pad to WAV_DURATION_S seconds
    target = int(round(WAV_DURATION_S * PCM_RATE))
    if len(pcm_f) >= target:
        pcm_f = pcm_f[:target]
    else:
        pcm_f = np.concatenate([pcm_f, np.zeros(target - len(pcm_f))])

    # Clip and convert to int32
    return np.clip(np.round(pcm_f * 32767), -32768, 32767).astype(np.int32)


def discover_wav_files(directory: Path, max_files: int) -> list:
    """Return up to max_files .wav paths found recursively under directory."""
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.wav"))[:max_files]

def make_chirp(n: int, rate: int = PCM_RATE) -> np.ndarray:
    """200 Hz -> 7 kHz linear chirp."""
    t   = np.arange(n) / rate
    dur = n / rate
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * 32767).astype(np.int32)


def make_silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int32)


def make_tone(n: int, freq: float = 1000.0, rate: int = PCM_RATE) -> np.ndarray:
    t    = np.arange(n) / rate
    win  = np.ones(n)
    fade = int(0.02 * n)
    win[:fade]  = np.hanning(2 * fade)[:fade]
    win[-fade:] = np.hanning(2 * fade)[fade:]
    return (np.sin(2 * np.pi * freq * t) * win * 32767).astype(np.int32)


def pcm_to_pdm(pcm: np.ndarray, decim: int = DECIM) -> np.ndarray:
    """First-order sigma-delta: each PCM sample -> decim PDM bits (0/1)."""
    pdm = []
    acc = 0
    for x in pcm:
        for _ in range(decim):
            acc += int(x)
            if acc >= 0:
                pdm.append(1)
                acc -= (1 << 15)
            else:
                pdm.append(0)
                acc += (1 << 15)
    return np.array(pdm, dtype=np.int8)

def _idle_flash(dut):
    dut.flash_mel_coeff_we_i.value   = 0
    dut.flash_mel_coeff_addr_i.value = 0
    dut.flash_mel_coeff_data_i.value = 0
    dut.flash_mel_index_we_i.value   = 0
    dut.flash_mel_index_addr_i.value = 0
    dut.flash_mel_index_data_i.value = 0
    dut.flash_log_lut_we_i.value     = 0
    dut.flash_log_lut_addr_i.value   = 0
    dut.flash_log_lut_data_i.value   = 0


async def flash_load_all(dut):
    _idle_flash(dut)

    with open(DATA_DIR / "mel_coeffs_sparse.hex") as f:
        coeffs = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d mel coeff entries..." % len(coeffs))
    dut.flash_mel_coeff_we_i.value = 1
    for addr, val in enumerate(coeffs):
        dut.flash_mel_coeff_addr_i.value = addr
        dut.flash_mel_coeff_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_coeff_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    with open(DATA_DIR / "mel_indices.hex") as f:
        indices = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d mel index entries..." % len(indices))
    dut.flash_mel_index_we_i.value = 1
    for addr, val in enumerate(indices):
        dut.flash_mel_index_addr_i.value = addr
        dut.flash_mel_index_data_i.value = val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_index_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    with open(DATA_DIR / "log2_lut.hex") as f:
        lut = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d log LUT entries..." % len(lut))
    dut.flash_log_lut_we_i.value = 1
    for addr, val in enumerate(lut):
        dut.flash_log_lut_addr_i.value = addr
        dut.flash_log_lut_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_log_lut_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    cocotb.log.info("All SRAMs loaded.")


async def do_reset(dut):
    dut.reset_i.value = 1
    dut.valid_i.value = 0
    dut.data_i.value  = 0
    _idle_flash(dut)
    await ClockCycles(dut.clk_i, 20)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 10)


async def drive_pdm(dut, pdm_bits: np.ndarray):
    for b in pdm_bits:
        dut.data_i.value  = 0x7FFF if b else -0x8000
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
    dut.valid_i.value = 0
    dut.data_i.value  = 0


async def collect_frames(dut, timeout: int) -> list:
    frames = []
    for _ in range(timeout):
        await RisingEdge(dut.clk_i)
        try:
            if int(dut.mel_compensated_valid_o.value):
                v = int(dut.mel_compensated_o.value)
                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])
                frames[-1].append(v)
        except (ValueError, AttributeError):
            pass
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
    return frames


def save_features(frames: list, name: str):
    if not frames:
        cocotb.log.info("No frames to save -- skipping %s" % name)
        return None
    mat = np.stack(
        [np.array(f, dtype=np.float32) / (1 << Q_FRAC) for f in frames],
        axis=1   # (N_MELS, n_frames)
    )
    path = os.path.join(os.path.dirname(__file__) or ".", name)
    np.save(path, mat)
    cocotb.log.info("Saved %s  shape=%s  range=[%.2f, %.2f] log2"
                    % (name, mat.shape, mat.min(), mat.max()))
    return mat


async def _run_pipeline(dut, pcm: np.ndarray, label: str,
                        npy_name: str, extra_checks=None):
    """Reset -> flash -> PDM -> collect -> save -> assert."""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)

    pdm = pcm_to_pdm(pcm)
    cocotb.log.info("[%s] %d PCM -> %d PDM bits" % (label, len(pcm), len(pdm)))

    ideal        = (len(pcm) - FFT_SIZE) // HOP + 1
    expected_min = max(1, ideal - 5)
    expected_max = ideal
    cocotb.log.info("[%s] Expected frames: %d - %d  (ideal=%d)"
                    % (label, expected_min, expected_max, ideal))

    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)

    cocotb.log.info("[%s] Frames produced: %d" % (label, len(frames)))
    mat = save_features(frames, npy_name)

    assert len(frames) > 0, "[%s] No frames produced" % label
    for i, f in enumerate(frames):
        assert len(f) == N_MELS, \
            "[%s] Frame %d has %d mels, expected %d" % (label, i, len(f), N_MELS)

    if extra_checks:
        extra_checks(frames, mat, label)

    cocotb.log.info("[%s] PASS -- %d frames" % (label, len(frames)))
    return frames, mat

# Test 1 -- Chirp

@cocotb.test()
async def test_full_pipeline_chirp(dut):
    """200 Hz -> 7 kHz swept chirp. Verifies chirp diagonal sweeps upward."""
    pcm = make_chirp(N_PCM_SAMPLES)

    def checks(frames, mat, label):
        if mat is None:
            return
        peak_bins  = np.argmax(mat, axis=0)
        n          = len(peak_bins)
        early_peak = float(np.mean(peak_bins[:n//4]))
        late_peak  = float(np.mean(peak_bins[3*n//4:]))
        cocotb.log.info("[%s] Chirp diagonal: early_bin=%.1f  late_bin=%.1f"
                        % (label, early_peak, late_peak))
        assert late_peak > early_peak + 3, (
            "[%s] Chirp diagonal not detected: early=%.1f late=%.1f"
            % (label, early_peak, late_peak)
        )

    await _run_pipeline(dut, pcm, "chirp", "rtl_features.npy", checks)

# Test 2 -- speech .wav files

@cocotb.test()
async def test_full_pipeline_speech(dut):
    """
    Runs real .wav files from SPEECH_WAV_DIR through the full RTL pipeline.

    Setup:
      Place your training-set .wav files in:
        <test_dir>/speech_data/
      or change SPEECH_WAV_DIR above to point at your dataset.

      Each file is:
        1. Loaded and resampled to 16 kHz
        2. Trimmed/padded to WAV_DURATION_S = 0.47 s
        3. Converted to PDM and driven through the RTL
        4. Saved as rtl_features_<filename>.npy

      If the directory does not exist or is empty, the test is skipped.
    """
    wav_files = discover_wav_files(SPEECH_WAV_DIR, MAX_WAV_FILES)

    if not wav_files:
        cocotb.log.info(
            "No .wav files found in %s -- skipping speech test.\n"
            "Create the directory and copy in your training .wav files:\n"
            "  mkdir -p %s\n"
            "  cp /your/dataset/yes/*.wav %s/"
            % (SPEECH_WAV_DIR, SPEECH_WAV_DIR, SPEECH_WAV_DIR)
        )
        return

    cocotb.log.info("Found %d .wav file(s) in %s" % (len(wav_files), SPEECH_WAV_DIR))

    # Flash SRAMs once before the loop
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)

    all_passed = True

    for wav_path in wav_files:
        # Safe label / filename (no spaces or long names)
        stem  = wav_path.stem.replace(" ", "_")[:40]
        label = "speech/%s" % stem
        npy   = "rtl_features_%s.npy" % stem

        await do_reset(dut)   # clean state between files

        # Load wav
        try:
            pcm = _load_wav(wav_path)
        except RuntimeError as e:
            cocotb.log.info("[%s] SKIP -- %s" % (label, e))
            continue

        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        cocotb.log.info("[%s] %s: %d samples  RMS=%.0f  peak=%d"
                        % (label, wav_path.name, len(pcm),
                           rms, int(np.abs(pcm).max())))

        if rms < 50:
            cocotb.log.info("[%s] WARNING: very low RMS -- file may be silent" % label)

        # Drive pipeline
        pdm     = pcm_to_pdm(pcm)
        timeout = len(pdm) + DRAIN
        cocotb.start_soon(drive_pdm(dut, pdm))
        frames = await collect_frames(dut, timeout)

        cocotb.log.info("[%s] Frames produced: %d" % (label, len(frames)))
        mat = save_features(frames, npy)

        # Assertions
        try:
            assert len(frames) > 0, "[%s] No frames produced" % label

            for i, f in enumerate(frames):
                assert len(f) == N_MELS, \
                    "[%s] Frame %d: %d mels, expected %d" % (label, i, len(f), N_MELS)

            if mat is not None and rms >= 50:
                mean_e = float(mat.mean())
                max_e  = float(mat.max())
                cocotb.log.info("[%s] Energy: mean=%.2f  max=%.2f log2"
                                % (label, mean_e, max_e))
                assert mean_e > 1.0, (
                    "[%s] Mean energy %.2f log2 too low -- signal lost"
                    % (label, mean_e)
                )
                assert max_e < 63.0, (
                    "[%s] Max energy %.2f log2 saturated" % (label, max_e)
                )

            cocotb.log.info("[%s] PASS" % label)

        except AssertionError as exc:
            cocotb.log.info("FAIL: %s" % exc)
            all_passed = False

    assert all_passed, "One or more speech files failed -- see log above"

# Test 3 -- Silence

@cocotb.test()
async def test_full_pipeline_silence(dut):
    """All-zero input. Verifies log floor and no saturation."""
    pcm = make_silence(N_PCM_SAMPLES)

    def checks(frames, mat, label):
        if mat is None or mat.shape[1] == 0:
            return
        mean_val = float(mat.mean())
        max_val  = float(mat.max())
        cocotb.log.info("[%s] Silence: mean=%.3f  max=%.3f log2"
                        % (label, mean_val, max_val))
        assert mean_val < 15.0, (
            "[%s] Silence mean %.2f log2 too high -- possible stuck pipeline"
            % (label, mean_val)
        )
        assert max_val < 60.0, (
            "[%s] Silence max %.2f log2 near saturation" % (label, max_val)
        )

    await _run_pipeline(dut, pcm, "silence", "rtl_features_silence.npy", checks)


# Test 4 -- Sustained 1 kHz tone

@cocotb.test()
async def test_full_pipeline_tone(dut):
    """1 kHz tone. Verifies energy in correct mel range, stable across frames."""
    TONE_FREQ = 1000.0
    pcm = make_tone(N_PCM_SAMPLES, freq=TONE_FREQ)

    def checks(frames, mat, label):
        if mat is None or mat.shape[1] == 0:
            return
        peak_bins = np.argmax(mat, axis=0)
        lo, hi    = 5, 20
        on_target = np.sum((peak_bins >= lo) & (peak_bins <= hi))
        cocotb.log.info(
            "[%s] Tone %.0f Hz: peak_bins=%s  on-target %d/%d mel [%d,%d]"
            % (label, TONE_FREQ, peak_bins[:8].tolist(),
               on_target, len(peak_bins), lo, hi)
        )
        assert on_target > len(peak_bins) * 0.7, (
            "[%s] %.0f Hz: only %d/%d frames in mel [%d,%d]"
            % (label, TONE_FREQ, on_target, len(peak_bins), lo, hi)
        )
        drift = int(peak_bins.max()) - int(peak_bins.min())
        cocotb.log.info("[%s] Tone peak bin drift: %d" % (label, drift))
        assert drift < 6, (
            "[%s] Tone peak bin drifts by %d -- pipeline unstable?" % (label, drift)
        )
        mean_val = float(mat.mean())
        cocotb.log.info("[%s] Tone mean energy: %.2f log2" % (label, mean_val))
        assert mean_val > 2.0, (
            "[%s] Tone mean %.2f log2 too low -- signal attenuated" % (label, mean_val)
        )

    await _run_pipeline(dut, pcm, "tone_1kHz", "rtl_features_tone.npy", checks)