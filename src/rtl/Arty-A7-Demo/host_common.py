#!/usr/bin/env python3
"""Shared host helpers for the Arty A7 KWS demos."""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np


ARTY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTY_ROOT.parents[2]
RTL_ROOT = REPO_ROOT / "src" / "rtl"
ML_ROOT = REPO_ROOT / "src" / "ml"
LOGMEL_DIR = RTL_ROOT / "Log-Mel" / "data"

DEFAULT_MODEL = "dscnn-24center-v1"
DEFAULT_MODEL_DIR = ML_ROOT / "models" / DEFAULT_MODEL
FULL_DEMO_DIR = ARTY_ROOT / "full_demo"
FULL_DEMO_HOST_DIR = FULL_DEMO_DIR / "host"

SYNC0 = 0xAA
SYNC1 = 0x55
ACK_BYTE = 0x06
NACK_BYTE = 0xEE
ERR_BYTE = 0xE1

MOD_FEATURES = 0x0
MOD_DSCNN = 0x1
MOD_AUDIO = 0x2
MOD_DEBUG = 0x8
MOD_CONTROL = 0xF

# Debug subtargets — spect_streamer.sv snoops these and walks the
# requested spectrogram bank back over UART after the boot_controller
# ACKs the request packet.
DBG_READ_SPECT_A = 0x0
DBG_READ_SPECT_B = 0x1

FEAT_LOG_LUT = 0x0
FEAT_MEL_COEFF = 0x1
FEAT_MEL_META = 0x2
FEAT_VAD_THRESH = 0x3  # 32-bit register, two 16-bit writes (low at addr=0, high at addr=1)
FEAT_INPUT_QUANT_MULT = 0x4  # same two-write 32-bit pattern; 0 = use RTL default

DSCNN_WEIGHTS = 0x0
DSCNN_CFG = 0x1
DSCNN_BIAS = 0x2  # 295 x INT32, byte-addressed little-endian (boot_pkg.sv)

CTRL_BOOT_DONE = 0x0
CTRL_SESSION_RESET = 0x2

CLASS_TAG_MASK = 0xF8
CLASS_TAG = 0xC0

SAMPLE_RATE = 16000
FFT_SIZE = 256
HOP = 128
START_FRAME = 37
N_FRAMES = 50
N_MELS = 40
SAMPLES_STREAM = FFT_SIZE + (START_FRAME + N_FRAMES - 1) * HOP


def make_target(mod: int, sub: int) -> int:
    return ((mod & 0xF) << 4) | (sub & 0xF)


def frame_packet(target: int, addr: int, payload: bytes) -> bytes:
    body = bytes([
        target & 0xFF,
        (addr >> 8) & 0xFF,
        addr & 0xFF,
        (len(payload) >> 8) & 0xFF,
        len(payload) & 0xFF,
    ]) + bytes(payload)
    cksum = 0
    for byte in body:
        cksum ^= byte
    return bytes([SYNC0, SYNC1]) + body + bytes([cksum & 0xFF])


def send_packet(ser, target: int, addr: int, payload: bytes,
                ack_timeout_s: float = 2.0) -> int:
    """Send one framed packet and return ACK/NACK/ERR or -1 on timeout."""
    ser.reset_input_buffer()
    ser.write(frame_packet(target, addr, payload))
    ser.flush()

    deadline = time.monotonic() + ack_timeout_s
    while time.monotonic() < deadline:
        data = ser.read(1)
        if not data:
            continue
        byte = data[0]
        if byte in (ACK_BYTE, NACK_BYTE, ERR_BYTE):
            return byte
        if (byte & CLASS_TAG_MASK) == CLASS_TAG:
            continue
    return -1


def response_name(resp: int) -> str:
    if resp == ACK_BYTE:
        return "ACK"
    if resp == NACK_BYTE:
        return "NACK"
    if resp == ERR_BYTE:
        return "ERR"
    if resp < 0:
        return "TIMEOUT"
    return f"0x{resp:02X}"


def pack_16bit_le(words) -> bytes:
    out = bytearray()
    for word in words:
        out.append(word & 0xFF)
        out.append((word >> 8) & 0xFF)
    return bytes(out)


def vad_threshold_payload(value: int) -> bytes:
    """4-byte FEAT_VAD_THRESH payload, sent at packet addr=0. The
    boot_controller's write_addr_q auto-increments, so the first 16-bit
    word lands at addr=0 (low half) and the second at addr=1 (high half),
    matching features_boot_router's split write."""
    value &= 0xFFFFFFFF
    return pack_16bit_le([value & 0xFFFF, (value >> 16) & 0xFFFF])


def parse_vad_arg(spec) -> int | None:
    """CLI helper: 'off' / None → None (skip write, register stays 0).
    'auto' → 0xFFFFFFFF (auto-calibration sentinel). Anything else is
    parsed as int (decimal or 0x-prefixed hex) and clamped to 32 bits."""
    if spec is None or spec == "off":
        return None
    if spec == "auto":
        return 0xFFFFFFFF
    value = int(spec, 0)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"vad threshold {value} out of 32-bit range")
    return value


def read_input_quant_mult(weights_path: Path) -> int | None:
    """Find input_quant.txt next to the given weights.hex and return
    `input_quant_mult` (the field after the keyword on its line) as int.
    Returns None if the file or the field isn't found — caller can fall
    back to skipping the write (RTL default takes effect)."""
    candidate = weights_path.parent / "input_quant.txt"
    if not candidate.exists():
        return None
    for line in candidate.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "input_quant_mult":
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def parse_input_quant_arg(spec, weights_path: Path | None = None) -> int | None:
    """CLI helper: 'off' / None → None (skip write, register stays 0,
    RTL falls back to its compile-time parameter default of 5817845).
    'auto' → read input_quant_mult from the input_quant.txt sitting
    next to weights_path; if missing, return None. Anything else is
    parsed as int (decimal or 0x-prefixed hex)."""
    if spec is None or spec == "off":
        return None
    if spec == "auto":
        if weights_path is None:
            return None
        return read_input_quant_mult(weights_path)
    value = int(spec, 0)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"input_quant_mult {value} out of 32-bit range")
    return value


def _hex_values(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            values.append(text.split()[0])
    return values


def load_hex16(path: Path) -> list[int]:
    return [int(value, 16) & 0xFFFF for value in _hex_values(path)]


def load_hex8(path: Path) -> list[int]:
    return [int(value, 16) & 0xFF for value in _hex_values(path)]


def stream_audio(ser, samples_int16, chunk_samples: int = 2048,
                 ack_timeout_s: float = 10.0) -> int:
    target = make_target(MOD_AUDIO, 0)
    sent = 0
    for offset in range(0, len(samples_int16), chunk_samples):
        payload = samples_int16[offset:offset + chunk_samples].astype("<i2").tobytes()
        resp = send_packet(ser, target, 0, payload, ack_timeout_s=ack_timeout_s)
        if resp != ACK_BYTE:
            print(f"audio chunk @ {offset}: {response_name(resp)}")
            return sent
        sent += 1
    return sent


def sine_samples(n_samples: int, freq_hz: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLE_RATE
    scale = amp * np.iinfo(np.int16).max
    return (scale * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)


def mic_samples(seconds: float = 1.0, n_required: int | None = None,
                device: int | str | None = None) -> np.ndarray:
    """Record from the system default microphone (or `device` index) at
    SAMPLE_RATE mono int16. Returns a 1-D np.int16. Truncates or pads to
    n_required if given. Blocks until recording completes."""
    import sounddevice as sd
    n_frames = int(round(seconds * SAMPLE_RATE))
    buf = sd.rec(n_frames, samplerate=SAMPLE_RATE, channels=1,
                 dtype="int16", device=device)
    sd.wait()
    samples = buf.flatten()
    if n_required is None:
        return samples
    if len(samples) < n_required:
        return np.pad(samples, (0, n_required - len(samples))).astype(np.int16)
    return samples[:n_required]


def decode_audio_file(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-supported audio container (m4a / mp3 / wav / ogg / …)
    to mono int16 at sample_rate. Returns 1-D np.int16. Used for offline
    continuous-mode testing with arbitrary recordings — wav_samples is
    only for the strict 16-bit / 16 kHz WAV path."""
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(sample_rate), "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype="<i2").copy()


def wav_samples(path: Path, n_required: int | None = None) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got sample width {width}")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")

    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, 0]
    if n_required is None:
        return samples
    if len(samples) < n_required:
        return np.pad(samples, (0, n_required - len(samples))).astype(np.int16)
    return samples[:n_required]
