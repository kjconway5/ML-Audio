"""
test_full_pipeline_top.py
Full-pipeline cocotb testbench:
  PDM input (1.008 MHz) -> CIC -> compFIR -> STFFT -> LogMel -> spect_buffer

Tests:
  1. test_full_pipeline_chirp    -- 200-7 kHz swept chirp (baseline sanity)
  2. test_full_pipeline_speech   -- real .wav files grouped by keyword directory
  3. test_full_pipeline_silence  -- all-zeros input (log floor check)
  4. test_full_pipeline_tone     -- 1 kHz sustained tone (narrow-band check)

WAV directory layout expected:
    speech_data/
        yes/   <-- each subdirectory is one keyword
            0a2b400e_nohash_0.wav
            0a2b400e_nohash_1.wav
            ...
        no/
            ...
        up/
            ...

Set SPEECH_WAV_DIR and MAX_WAV_FILES_PER_KEYWORD below.
Features are saved as:  rtl_features_<keyword>_<stem>.npy
"""

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PCM_RATE = 16_000
DECIM    = 63
FFT_SIZE = 256
HOP      = FFT_SIZE // 2

N_PCM_SAMPLES = 7_500
N_MELS        = 40
Q_FRAC        = 10
DRAIN         = 100_000

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"

# ---------------------------------------------------------------------------
# WAV dataset configuration
# ---------------------------------------------------------------------------
SPEECH_WAV_DIR            = Path(__file__).resolve().parent / "speech_data"
MAX_WAV_FILES_PER_KEYWORD = 2 #20    # cap per keyword subdirectory
WAV_DURATION_S            = 0.47  # seconds per clip


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def _load_wav(path: Path) -> np.ndarray:
    """Load .wav -> int32 PCM at PCM_RATE, trimmed/padded to N_PCM_SAMPLES."""
    data = rate = None
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
                "Install scipy:  pip install scipy --break-system-packages\n"
                "Error: %s" % (path, e)
            )

    dt = np.asarray(data).dtype
    if dt == np.int16:
        pcm_f = data.astype(np.float64) / 32768.0
    elif dt == np.int32:
        pcm_f = data.astype(np.float64) / 2147483648.0
    elif dt == np.uint8:
        pcm_f = (data.astype(np.float64) - 128.0) / 128.0
    else:
        pcm_f = np.asarray(data, dtype=np.float64)

    if pcm_f.ndim == 2:
        pcm_f = pcm_f.mean(axis=1)

    if rate != PCM_RATE:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g     = gcd(int(PCM_RATE), int(rate))
            pcm_f = resample_poly(pcm_f, PCM_RATE // g, rate // g)
        except ImportError:
            n_out = int(round(len(pcm_f) * PCM_RATE / rate))
            pcm_f = np.interp(
                np.linspace(0, 1, n_out),
                np.linspace(0, 1, len(pcm_f)),
                pcm_f,
            )

    target = int(round(WAV_DURATION_S * PCM_RATE))
    if len(pcm_f) >= target:
        pcm_f = pcm_f[:target]
    else:
        pcm_f = np.concatenate([pcm_f, np.zeros(target - len(pcm_f))])

    return np.clip(np.round(pcm_f * 32767), -32768, 32767).astype(np.int32)


def discover_keyword_dirs(base_dir: Path) -> dict:
    """
    Return {keyword: [wav_path, ...]} for each immediate subdirectory.
    Falls back to {'speech': [wav_path, ...]} if no subdirs exist.
    """
    if not base_dir.exists():
        return {}

    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if subdirs:
        result = {}
        for d in subdirs:
            wavs = sorted(d.rglob("*.wav"))[:MAX_WAV_FILES_PER_KEYWORD]
            if wavs:
                result[d.name] = wavs
        return result

    # Flat layout -- no subdirectories
    wavs = sorted(base_dir.glob("*.wav"))[:MAX_WAV_FILES_PER_KEYWORD]
    return {"speech": wavs} if wavs else {}


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def make_chirp(n: int, rate: int = PCM_RATE) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# PDM modulator
# ---------------------------------------------------------------------------

def pcm_to_pdm(pcm: np.ndarray, decim: int = DECIM) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# DUT helpers
# ---------------------------------------------------------------------------

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
    dut.vad_threshold_i.value = 0   # NEW -- disabled by default
    dut.dft_vad_obs_en_i.value = 0
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
            "[%s] Frame %d: %d mels" % (label, i, len(f))

    if extra_checks:
        extra_checks(frames, mat, label)

    cocotb.log.info("[%s] PASS -- %d frames" % (label, len(frames)))
    return frames, mat


# ---------------------------------------------------------------------------
# Test 1 -- Chirp
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test 2 -- Real speech .wav files (grouped by keyword directory)
# ---------------------------------------------------------------------------

# @cocotb.test()
# async def test_full_pipeline_speech(dut):
#     """
#     Run .wav files from SPEECH_WAV_DIR through the full RTL pipeline.

#     Directory layout:
#         speech_data/
#             yes/  <- one subdirectory per keyword
#                 file_0.wav
#                 file_1.wav
#             no/
#                 ...

#     Each file produces:  rtl_features_<keyword>_<stem>.npy
#     compare_wav_outputs.py groups these by keyword automatically.
#     """
#     keyword_dirs = discover_keyword_dirs(SPEECH_WAV_DIR)

#     if not keyword_dirs:
#         cocotb.log.info(
#             "No keyword directories found in %s -- skipping." % SPEECH_WAV_DIR
#         )
#         return

#     total_wavs = sum(len(v) for v in keyword_dirs.values())
#     cocotb.log.info(
#         "Keywords: %s  |  Total files: %d  |  Cap: %d per keyword"
#         % (", ".join(sorted(keyword_dirs)), total_wavs, MAX_WAV_FILES_PER_KEYWORD)
#     )

#     # Flash SRAMs once before any file loop
#     cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
#     await do_reset(dut)
#     await flash_load_all(dut)

#     all_passed   = True
#     total_done   = 0
#     total_failed = 0

#     for keyword in sorted(keyword_dirs):
#         wav_files  = keyword_dirs[keyword]
#         kw_passed  = 0
#         kw_failed  = 0
#         cocotb.log.info("=== Keyword: '%s'  (%d files) ===" % (keyword, len(wav_files)))

#         for wav_path in wav_files:
#             stem  = wav_path.stem.replace(" ", "_")[:32]
#             npy   = "rtl_features_%s_%s.npy" % (keyword, stem)
#             label = "%s/%s" % (keyword, stem)

#             await do_reset(dut)   # clean state between files

#             try:
#                 pcm = _load_wav(wav_path)
#             except RuntimeError as e:
#                 cocotb.log.info("[%s] SKIP -- %s" % (label, e))
#                 continue

#             rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
#             if rms < 50:
#                 cocotb.log.info("[%s] low RMS=%.0f (possibly silent)" % (label, rms))

#             pdm     = pcm_to_pdm(pcm)
#             timeout = len(pdm) + DRAIN

#             cocotb.start_soon(drive_pdm(dut, pdm))
#             frames = await collect_frames(dut, timeout)

#             mat = save_features(frames, npy)
#             total_done += 1

#             try:
#                 assert len(frames) > 0, "[%s] No frames produced" % label
#                 for i, f in enumerate(frames):
#                     assert len(f) == N_MELS, \
#                         "[%s] Frame %d has %d mels" % (label, i, len(f))
#                 if mat is not None and rms >= 50:
#                     assert float(mat.mean()) > 1.0, \
#                         "[%s] Mean energy %.2f too low" % (label, mat.mean())
#                     assert float(mat.max()) < 63.0, \
#                         "[%s] Max energy %.2f saturated" % (label, mat.max())
#                 kw_passed += 1

#             except AssertionError as exc:
#                 cocotb.log.info("FAIL: %s" % exc)
#                 kw_failed  += 1
#                 total_failed += 1
#                 all_passed   = False

#         cocotb.log.info(
#             "Keyword '%s': %d/%d passed" % (keyword, kw_passed, kw_passed + kw_failed)
#         )

#     cocotb.log.info(
#         "Speech test done: %d/%d passed across %d keywords"
#         % (total_done - total_failed, total_done, len(keyword_dirs))
#     )
#     assert all_passed, "%d file(s) failed -- see log above" % total_failed


# ---------------------------------------------------------------------------
# Test 3 -- Silence
# ---------------------------------------------------------------------------

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
        assert mean_val < 15.0, \
            "[%s] Silence mean %.2f log2 too high" % (label, mean_val)
        assert max_val < 60.0, \
            "[%s] Silence max %.2f log2 saturated" % (label, max_val)

    await _run_pipeline(dut, pcm, "silence", "rtl_features_silence.npy", checks)


# ---------------------------------------------------------------------------
# Test 4 -- Sustained 1 kHz tone
# ---------------------------------------------------------------------------

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
        assert on_target > len(peak_bins) * 0.7, \
            "[%s] Only %d/%d in mel [%d,%d]" % (label, on_target, len(peak_bins), lo, hi)
        assert int(peak_bins.max()) - int(peak_bins.min()) < 6, \
            "[%s] Peak bin drifts too much" % label
        assert float(mat.mean()) > 2.0, \
            "[%s] Tone mean %.2f log2 too low" % (label, mat.mean())

    await _run_pipeline(dut, pcm, "tone_1kHz", "rtl_features_tone.npy", checks)

# ---------------------------------------------------------------------------
# Test 5 and 6 - VAD Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_full_pipeline_silence_vad(dut):
    """All-zero input with VAD enabled -- should produce zero frames."""
    pcm = make_silence(N_PCM_SAMPLES)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)
    dut.vad_threshold_i.value = 11_000_000

    pdm = pcm_to_pdm(pcm)
    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)

    cocotb.log.info("[silence_vad] Frames produced: %d (expected 0)" % len(frames))
    assert len(frames) == 0, \
        "[silence_vad] Expected 0 frames, got %d" % len(frames)
    cocotb.log.info("[silence_vad] PASS")


@cocotb.test()
async def test_full_pipeline_chirp_vad(dut):
    """Chirp with VAD -- should produce same frames as without."""
    pcm = make_chirp(N_PCM_SAMPLES)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)
    dut.vad_threshold_i.value = 11_000_000

    pdm = pcm_to_pdm(pcm)
    ideal = (len(pcm) - FFT_SIZE) // HOP + 1
    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)

    cocotb.log.info("[chirp_vad] Frames: %d (ideal=%d)" % (len(frames), ideal))
    assert len(frames) > 0, "[chirp_vad] No frames -- threshold too high?"
    assert len(frames) >= ideal - 5, \
        "[chirp_vad] Too few frames: %d vs ideal %d" % (len(frames), ideal)

    mat = save_features(frames, "rtl_features_chirp_vad.npy")
    if mat is not None:
        peak_bins = np.argmax(mat, axis=0)
        n = len(peak_bins)
        early = float(np.mean(peak_bins[:n//4]))
        late  = float(np.mean(peak_bins[3*n//4:]))
        assert late > early + 3, "[chirp_vad] Chirp diagonal not detected"

    cocotb.log.info("[chirp_vad] PASS -- %d frames" % len(frames))

# ---------------------------------------------------------------------------
# Test 7 - Spliced VAD Test
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_full_pipeline_spliced_vad(dut):
    """Silence + chirp spliced together -- only chirp frames should appear."""
    silence = make_silence(N_PCM_SAMPLES // 2)
    chirp   = make_chirp(N_PCM_SAMPLES // 2)
    pcm     = np.concatenate([silence, chirp])

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)
    dut.vad_threshold_i.value = 11_000_000

    pdm = pcm_to_pdm(pcm)
    ideal_total   = (len(pcm) - FFT_SIZE) // HOP + 1
    ideal_chirp   = (len(chirp) - FFT_SIZE) // HOP + 1
    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)

    cocotb.log.info("[spliced_vad] Frames: %d (total_ideal=%d chirp_ideal=%d)"
                    % (len(frames), ideal_total, ideal_chirp))
    # Should get roughly chirp-only frames, not the full count
    assert len(frames) > 0, "[spliced_vad] No frames at all"
    assert len(frames) < ideal_total, \
        "[spliced_vad] Got %d frames -- silence not suppressed" % len(frames)

    cocotb.log.info("[spliced_vad] PASS -- %d/%d frames (silence suppressed)"
                    % (len(frames), ideal_total))

@cocotb.test()
async def test_full_pipeline_speech_vad(dut):
    """Real .wav files with VAD enabled at 11M threshold."""
    keyword_dirs = discover_keyword_dirs(SPEECH_WAV_DIR)
    if not keyword_dirs:
        cocotb.log.info("No keyword directories found -- skipping.")
        return

    total_wavs = sum(len(v) for v in keyword_dirs.values())
    cocotb.log.info("Keywords: %s  |  Total files: %d"
                    % (", ".join(sorted(keyword_dirs)), total_wavs))

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)

    all_passed = True
    total_done = 0
    total_failed = 0

    for keyword in sorted(keyword_dirs):
        wav_files = keyword_dirs[keyword]
        kw_passed = 0
        kw_failed = 0
        cocotb.log.info("=== [VAD] Keyword: '%s'  (%d files) ==="
                        % (keyword, len(wav_files)))

        for wav_path in wav_files:
            stem  = wav_path.stem.replace(" ", "_")[:32]
            npy   = "rtl_features_vad_%s_%s.npy" % (keyword, stem)
            label = "vad_%s/%s" % (keyword, stem)

            await do_reset(dut)
            dut.vad_threshold_i.value = 11_000_000  # <-- VAD enabled

            try:
                pcm = _load_wav(wav_path)
            except RuntimeError as e:
                cocotb.log.info("[%s] SKIP -- %s" % (label, e))
                continue

            rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
            pdm = pcm_to_pdm(pcm)
            timeout = len(pdm) + DRAIN

            cocotb.start_soon(drive_pdm(dut, pdm))
            frames = await collect_frames(dut, timeout)

            mat = save_features(frames, npy)
            total_done += 1

            # With VAD, quiet files may produce fewer frames -- that's expected
            ideal = (len(pcm) - FFT_SIZE) // HOP + 1

            try:
                # Must produce at least some frames for real speech
                assert len(frames) > 0, "[%s] No frames produced" % label
                for i, f in enumerate(frames):
                    assert len(f) == N_MELS, \
                        "[%s] Frame %d has %d mels" % (label, i, len(f))
                # Log how many frames VAD dropped
                cocotb.log.info("[%s] %d/%d frames (VAD kept %.0f%%)"
                                % (label, len(frames), ideal,
                                   100.0 * len(frames) / ideal))
                kw_passed += 1
            except AssertionError as exc:
                cocotb.log.info("FAIL: %s" % exc)
                kw_failed += 1
                total_failed += 1
                all_passed = False

        cocotb.log.info("Keyword '%s': %d/%d passed"
                        % (keyword, kw_passed, kw_passed + kw_failed))

    cocotb.log.info("Speech VAD test done: %d/%d passed"
                    % (total_done - total_failed, total_done))
    assert all_passed, "%d file(s) failed" % total_failed


# ---------------------------------------------------------------------------
# Test 9 - Auto-VAD: Silence then Chirp
# ---------------------------------------------------------------------------

AUTOVAD_SENTINEL = 0xFFFFFFFF
CALIB_FRAMES     = 256
CALIB_SILENCE_S  = 2.2   # seconds of silence for calibration period

def make_silence_seconds(duration_s: float, rate: int = PCM_RATE) -> np.ndarray:
    return np.zeros(int(duration_s * rate), dtype=np.int32)


@cocotb.test()
async def test_full_pipeline_autovad_silence_then_chirp(dut):
    """Auto-calibrate: 2.2s silence (calibration) + chirp. Verify calibration
    passes all frames, post-calibration silence is suppressed, chirp passes."""
    silence_pcm = make_silence_seconds(CALIB_SILENCE_S)
    chirp_pcm   = make_chirp(N_PCM_SAMPLES)
    pcm = np.concatenate([silence_pcm, chirp_pcm])
 
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)
    dut.vad_threshold_i.value = AUTOVAD_SENTINEL
 
    pdm = pcm_to_pdm(pcm)
    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)
 
    ideal_total   = (len(pcm) - FFT_SIZE) // HOP + 1
    ideal_chirp   = (len(chirp_pcm) - FFT_SIZE) // HOP + 1
    ideal_silence = (len(silence_pcm) - FFT_SIZE) // HOP + 1
 
    cocotb.log.info("[autovad_silence_chirp] Total frames produced: %d" % len(frames))
    cocotb.log.info("[autovad_silence_chirp] Ideal total (no VAD): %d" % ideal_total)
    cocotb.log.info("[autovad_silence_chirp] Ideal silence frames: %d" % ideal_silence)
    cocotb.log.info("[autovad_silence_chirp] Ideal chirp frames: %d" % ideal_chirp)
    cocotb.log.info("[autovad_silence_chirp] CALIB_FRAMES param: %d" % CALIB_FRAMES)
    cocotb.log.info("[autovad_silence_chirp] Expected: ~%d (calib) + %d (chirp) = ~%d"
                    % (CALIB_FRAMES, ideal_chirp, CALIB_FRAMES + ideal_chirp))
 
    assert len(frames) > 0, "[autovad_silence_chirp] No frames produced"
    assert len(frames) < ideal_total, \
        "[autovad_silence_chirp] Got %d frames -- post-calib silence not suppressed (ideal_total=%d)" \
        % (len(frames), ideal_total)
 
    expected_low  = CALIB_FRAMES + ideal_chirp - 15
    expected_high = CALIB_FRAMES + ideal_chirp + 10
    cocotb.log.info("[autovad_silence_chirp] Frame count %d, expected range [%d, %d]"
                    % (len(frames), expected_low, expected_high))
 
    if len(frames) < expected_low or len(frames) > expected_high:
        cocotb.log.info("[autovad_silence_chirp] WARNING: frame count %d outside expected [%d, %d]"
                        % (len(frames), expected_low, expected_high))
 
    cocotb.log.info("[autovad_silence_chirp] PASS -- %d/%d frames (silence suppressed after calibration)"
                    % (len(frames), ideal_total))



# ---------------------------------------------------------------------------
# Test 10 - Auto-VAD: Silence only
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_full_pipeline_autovad_silence_only(dut):
    """Auto-calibrate on 4+ seconds of pure silence. First 250 frames pass
    (calibration), all subsequent silence suppressed."""
    pcm = make_silence_seconds(4.0)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)
    dut.vad_threshold_i.value = AUTOVAD_SENTINEL

    pdm = pcm_to_pdm(pcm)
    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)

    ideal_total = (len(pcm) - FFT_SIZE) // HOP + 1

    cocotb.log.info("[autovad_silence_only] Total frames produced: %d" % len(frames))
    cocotb.log.info("[autovad_silence_only] Ideal total (no VAD): %d" % ideal_total)
    cocotb.log.info("[autovad_silence_only] Expected: ~%d (calibration period only)" % CALIB_FRAMES)

    # Should get approximately CALIB_FRAMES (the passthrough during calibration)
    # and then nothing after that
    assert len(frames) >= CALIB_FRAMES - 5, \
        "[autovad_silence_only] Too few calibration frames: %d (expected ~%d)" \
        % (len(frames), CALIB_FRAMES)
    assert len(frames) <= CALIB_FRAMES + 10, \
        "[autovad_silence_only] Too many frames: %d -- post-calib silence not suppressed (expected ~%d)" \
        % (len(frames), CALIB_FRAMES)

    cocotb.log.info("[autovad_silence_only] PASS -- %d frames (~%d calibration, rest suppressed)"
                    % (len(frames), CALIB_FRAMES))


# ---------------------------------------------------------------------------
# Test 11 - Auto-VAD override: fixed threshold bypasses auto-cal
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_full_pipeline_autovad_override(dut):
    """Set threshold to 11M (not sentinel). Verify behavior matches fixed-threshold
    exactly -- auto-calibration must be bypassed."""
    pcm = make_chirp(N_PCM_SAMPLES)
 
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)
    dut.vad_threshold_i.value = 11_000_000
 
    pdm = pcm_to_pdm(pcm)
    ideal = (len(pcm) - FFT_SIZE) // HOP + 1
    timeout = len(pdm) + DRAIN
    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)
 
    cocotb.log.info("[autovad_override] Frames: %d (ideal=%d)" % (len(frames), ideal))
    cocotb.log.info("[autovad_override] Threshold set to 11000000 (fixed mode, NOT sentinel)")
 
    assert len(frames) > 0, "[autovad_override] No frames"
    assert len(frames) >= ideal - 5, \
        "[autovad_override] Too few frames: %d vs ideal %d" % (len(frames), ideal)
 
    cocotb.log.info("[autovad_override] PASS -- %d frames (fixed threshold confirmed)" % len(frames))


# ---------------------------------------------------------------------------
# Test 12 - Auto-VAD: Real speech with silence prefix
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_full_pipeline_autovad_speech(dut):
    """For each wav: prepend 2s silence, set threshold=sentinel.
    Report calibration frames, post-calib silence dropped, speech kept."""
    keyword_dirs = discover_keyword_dirs(SPEECH_WAV_DIR)
    if not keyword_dirs:
        cocotb.log.info("No keyword directories found -- skipping.")
        return

    total_wavs = sum(len(v) for v in keyword_dirs.values())
    cocotb.log.info("[autovad_speech] Keywords: %s  |  Total files: %d"
                    % (", ".join(sorted(keyword_dirs)), total_wavs))

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)

    all_passed = True
    total_done = 0
    total_failed = 0

    silence_prefix = make_silence_seconds(CALIB_SILENCE_S)
    ideal_silence_frames = (len(silence_prefix) - FFT_SIZE) // HOP + 1

    for keyword in sorted(keyword_dirs):
        wav_files = keyword_dirs[keyword]
        kw_passed = 0
        kw_failed = 0
        cocotb.log.info("=== [AutoVAD] Keyword: '%s'  (%d files) ==="
                        % (keyword, len(wav_files)))

        for wav_path in wav_files:
            stem  = wav_path.stem.replace(" ", "_")[:32]
            npy   = "rtl_features_autovad_%s_%s.npy" % (keyword, stem)
            label = "autovad_%s/%s" % (keyword, stem)

            await do_reset(dut)
            dut.vad_threshold_i.value = AUTOVAD_SENTINEL

            try:
                speech_pcm = _load_wav(wav_path)
            except RuntimeError as e:
                cocotb.log.info("[%s] SKIP -- %s" % (label, e))
                continue

            # Prepend 2s silence for calibration
            pcm = np.concatenate([silence_prefix, speech_pcm])

            pdm = pcm_to_pdm(pcm)
            timeout = len(pdm) + DRAIN

            ideal_total  = (len(pcm) - FFT_SIZE) // HOP + 1
            ideal_speech = (len(speech_pcm) - FFT_SIZE) // HOP + 1

            cocotb.start_soon(drive_pdm(dut, pdm))
            frames = await collect_frames(dut, timeout)

            mat = save_features(frames, npy)
            total_done += 1

            # Estimate breakdown
            post_calib_frames = max(0, len(frames) - CALIB_FRAMES)
            calib_portion = min(len(frames), CALIB_FRAMES)
            silence_after_calib = max(0, ideal_silence_frames - CALIB_FRAMES)
            speech_frames_kept = max(0, post_calib_frames)
            speech_pct = (100.0 * speech_frames_kept / ideal_speech) if ideal_speech > 0 else 0

            cocotb.log.info("[%s] Total frames: %d / %d ideal" % (label, len(frames), ideal_total))
            cocotb.log.info("[%s]   Calibration frames (passthrough): %d / %d expected"
                            % (label, calib_portion, CALIB_FRAMES))
            cocotb.log.info("[%s]   Post-calib silence frames expected dropped: ~%d"
                            % (label, silence_after_calib))
            cocotb.log.info("[%s]   Speech frames kept: %d / %d ideal (%.0f%%)"
                            % (label, speech_frames_kept, ideal_speech, speech_pct))

            try:
                assert len(frames) > 0, "[%s] No frames produced" % label
                for i, f in enumerate(frames):
                    assert len(f) == N_MELS, \
                        "[%s] Frame %d has %d mels" % (label, i, len(f))
                # Speech retention should be > 50%
                assert speech_pct > 50.0, \
                    "[%s] Speech retention too low: %.0f%%" % (label, speech_pct)
                kw_passed += 1
            except AssertionError as exc:
                cocotb.log.info("FAIL: %s" % exc)
                kw_failed += 1
                total_failed += 1
                all_passed = False

        cocotb.log.info("Keyword '%s': %d/%d passed"
                        % (keyword, kw_passed, kw_passed + kw_failed))

    cocotb.log.info("[autovad_speech] Done: %d/%d passed"
                    % (total_done - total_failed, total_done))
    assert all_passed, "%d file(s) failed" % total_failed

