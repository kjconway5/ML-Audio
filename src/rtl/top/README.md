# ML-Audio DSP Top

Top-level integration and verification for the ML-Audio keyword-spotting audio front-end, targeting GF180MCU via the wafer.space shuttle. This directory stitches the per-block RTL (CIC, FIR, STFFT, Log-Mel, spectrogram buffer) into two end-to-end pipelines and verifies them against a bit-accurate Python golden model and a torchaudio floating-point reference.

```
PDM (1.008 MHz) → CIC ÷63 → compFIR → STFFT (256-pt, 50% overlap) → Log-Mel (40 bins) → spectrogram buffer → DS-CNN
                  └──────── full_pipeline_top only ────────┘
PCM (16 kHz) ────────────────────────────→ STFFT → Log-Mel → spectrogram buffer
                  └──────────────── pipeline_top ─────────────────┘
```

## DSP Block & Feature Extraction

| | `pipeline_top.sv` | `full_pipeline_top.sv` |
|---|---|---|
| Input | 16-bit PCM @ 16 kHz | 1-bit PDM @ 1.008 MHz (as ±full-scale PCM) |
| Front end | — | CIC decimator (R=63, N=3) + compFIR compensation |
| Backpressure | `stfft.i_ready` (hierarchical probe) | `ready_o` top-level port (CIC propagates `i_ready` upstream) |
| VAD | — | Fixed threshold, auto-calibrating (sentinel `0xFFFFFFFF`), or disabled (0) via `vad_threshold_i`; `vad_active_o` + frame-drop observability |
| Use | STFFT + Log-Mel integration debug | Full-chip datapath verification |

Both instantiate the same downstream chain: the single-channel ready/valid `stfft`, frame-sync/bin-count alignment logic, `logmel_top`, BFP exponent compensation, and `spect_buffer_ctrl`.

## Directory contents

```
.
├── pipeline_top.sv             PCM-input top (STFFT → LogMel → spect buf)
├── full_pipeline_top.sv        PDM-input top (CIC → FIR → STFFT → LogMel → spect buf)
├── Makefile                    cocotb/verilator build for both tops
├── test_pipeline_top.py        cocotb tests for pipeline_top (chirp)
├── test_full_pipeline_top.py   cocotb tests for full_pipeline_top
│                                 (chirp / speech WAVs / silence / tone, VAD tests stubbed)
├── compare_outputs.py          RTL vs golden vs torchaudio (chirp), 4-panel plot
├── compare_wav_outputs.py      RTL vs golden per keyword (speech), accuracy summary
└── speech_data/                WAV dataset, one subdirectory per keyword
    ├── yes/*.wav
    └── no/*.wav
```

Block RTL lives in sibling directories referenced by the Makefile: `../CIC`, `../FIR`, `../STFFT` (R2FFT core + stfft wrapper + GF180 SRAM model), `../Log-Mel` (RTL, IP, and hex data), `../SPECT_BUFFER`.

## Running the tests

Requirements: `Verilator, cocotb, numpy, scipy, matplotlib`

```bash
# PCM-input pipeline (pipeline_top)
make test-pipeline

# Full PDM-input pipeline (full_pipeline_top)
make test-full

# Clean build artifacts, hex symlinks, and result files
make clean
```

`link-hex` (run automatically) symlinks the required `$readmemh` data into the build directory: Hanning window, mel coefficient/index tables, log2 LUT, and the R2FFT twiddle ROM hex files.

### Test suites

`test_pipeline_top.py` (PCM, CE_EVERY = 96 clocks/sample):

| Test | Checks |
|---|---|
| `test_frames` | STFFT emits `o_last` pulses; Log-Mel produces ≥ 1 frame |
| `test_pipeline` | Frame count in range, 40 mels/frame, values in range, saves diagnostics |

`test_full_pipeline_top.py` (PDM @ 1 bit/clock, 16 MHz sim clock):

*Baseline (VAD disabled, `vad_threshold_i = 0`):*

| Test | Stimulus | Checks |
|---|---|---|
| `test_full_pipeline_chirp` | 200 Hz → 7 kHz sweep | Chirp diagonal ascends across frames |
| `test_full_pipeline_speech` | WAVs from `speech_data/<keyword>/` | Frame shape, energy floor/ceiling per file |
| `test_full_pipeline_silence` | All zeros | Log floor < 15 log2, no saturation |
| `test_full_pipeline_tone` | 1 kHz tone | Peak in mel 5–20, stable, adequate energy |

*Fixed-threshold VAD (`vad_threshold_i = 11_000_000`):*

| Test | Stimulus | Checks |
|---|---|---|
| `test_full_pipeline_silence_vad` | All zeros | Zero frames emitted (all suppressed) |
| `test_full_pipeline_chirp_vad` | Chirp | Frame count ≥ ideal − 5; diagonal intact (loud input unaffected) |
| `test_full_pipeline_spliced_vad` | ½ silence + ½ chirp | Fewer than total ideal frames (silence portion suppressed), chirp frames present |
| `test_full_pipeline_speech_vad` | WAVs per keyword | Frames produced for real speech; logs VAD retention % per file |

*Auto-calibrating VAD (`vad_threshold_i = 0xFFFFFFFF` sentinel, `CALIB_FRAMES = 256`):*

| Test | Stimulus | Checks |
|---|---|---|
| `test_full_pipeline_autovad_silence_then_chirp` | 2.2 s silence + chirp | ~CALIB_FRAMES pass during calibration, post-calib silence suppressed, chirp passes |
| `test_full_pipeline_autovad_silence_only` | 4 s silence | Frame count ≈ CALIB_FRAMES (calibration passthrough only) |
| `test_full_pipeline_autovad_override` | Chirp, threshold = 11M | Behaves exactly like fixed-threshold mode (auto-cal bypassed) |
| `test_full_pipeline_autovad_speech` | 2.2 s silence prefix + WAVs | Calibration passthrough, then speech retention > 50% |

VAD semantics: each frame's spectral energy is compared against the threshold inside `logmel_top` (`spectral_vad`); below-threshold frames are dropped before the output buffer, `vad_active_o` reflects detection state, and `vad_frame_drop_ol` pulses on each dropped frame (observable when `dft_vad_obs_en_i` is set). Writing the `0xFFFFFFFF` sentinel instead of a threshold enables auto-calibration: the first `CALIB_FRAMES` frames pass through unconditionally while the block learns the noise floor, after which the derived threshold takes effect. Any other non-zero value is used directly as a fixed threshold; zero disables VAD entirely.

The PDM driver **must** honor `ready_o` — `drive_pdm` holds `valid_i` until the handshake completes. The ideal frame count for an `N`-sample clip is `(N − 256) // 128 + 1` (57 for the 7500-sample chirp). VAD tests intentionally produce fewer frames than ideal; the assertions account for this.

### Analysis scripts

```bash
# Chirp: RTL vs golden vs torchaudio, saves comparison.png
python3 compare_outputs.py [--rtl rtl_features.npy]

# Speech: per-keyword accuracy table + spectrogram plots + bar chart
python3 compare_wav_outputs.py [--rtl_dir .] [--wav_dir speech_data] [--vad_threshold N]
```

The golden model (`golden_model_spectral_vad.FullPipelineGoldenExtractor`, in `../../ml/Pipeline`) is a bit-accurate fixed-point replica of the RTL chain including per-frame BFP and exponent compensation; torchaudio (`features.py`) is the floating-point reference the DS-CNN was trained against. When comparing a VAD-enabled RTL run, pass the same threshold to the golden via `--vad_threshold` so both sides drop the same frames — otherwise the frame streams misalign and every downstream delta is meaningless.

### Output files

| File | Contents |
|---|---|
| `rtl_features*.npy` | (40, n_frames) log2 mel features, Q10 → float |
| `rtl_features_precomp.npy` | Pre-BFP-compensation mel output |
| `rtl_bfpexps.npy` | Per-frame BFP exponent (int8) |
| `rtl_fft_re.npy`, `rtl_fft_im.npy` | Raw 129-bin FFT dumps per frame |
| `rtl_sync_sample_counts.npy` | Input-sample count at each frame sync (deltas should be 128) |
| `comparison.png`, `spectrogram_<kw>.png`, `accuracy_summary.png` | Plots |

## Parameters

| Parameter | Value | Where |
|---|---|---|
| `FFT_SIZE` / `HOP` | 256 / 128 (50% overlap) | stfft, both tops |
| `N_BINS` / `N_MELS` | 129 / 40 | logmel |
| `CIC_RMAX` / `CIC_STAGES` | 63 / 3 (1.008 MHz → 16 kHz) | full top |
| `FIR_NTAPS` | 33 (CIC droop compensation) | full top |
| `BFP_Q_FRAC` | 10 (log2 output is Q6.10) | both tops |
| `START_FRAME` / `N_FRAMES` | 37 / 50 (spectrogram capture window) | spect buffer |
| `CE_EVERY` | 96 clocks/sample minimum for PCM tests | test_pipeline_top.py |
| `VAD threshold` | 11,000,000 (fixed) / `0xFFFFFFFF` (auto-cal) / 0 (off) | test_full_pipeline_top.py |
| `CALIB_FRAMES` | 256 (auto-VAD calibration passthrough) | test_full_pipeline_top.py |

### Authors:
* Michael Aguero: @mbaguero
* Aydan Olaez: @aolaez

