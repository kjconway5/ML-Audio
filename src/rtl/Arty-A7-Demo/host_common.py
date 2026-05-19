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
MOD_CONTROL = 0xF

FEAT_LOG_LUT = 0x0
FEAT_MEL_COEFF = 0x1
FEAT_MEL_META = 0x2

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
