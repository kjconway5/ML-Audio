#!/usr/bin/env python3
"""Host driver for full_demo_top."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from host_common import (  # noqa: E402
    ACK_BYTE,
    CLASS_TAG,
    CLASS_TAG_MASK,
    CTRL_BOOT_DONE,
    CTRL_SESSION_RESET,
    DEFAULT_MODEL_DIR,
    DSCNN_BIAS,
    DSCNN_CFG,
    DSCNN_WEIGHTS,
    FEAT_LOG_LUT,
    FEAT_MEL_COEFF,
    FEAT_MEL_META,
    FULL_DEMO_HOST_DIR,
    LOGMEL_DIR,
    MOD_CONTROL,
    MOD_DSCNN,
    MOD_FEATURES,
    SAMPLES_STREAM,
    make_target,
    pack_16bit_le,
    response_name,
    send_packet,
    sine_samples,
    stream_audio,
    load_hex16,
    load_hex8,
    wav_samples,
)


def load_cfg_pairs(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            addr, value = text.split()[:2]
            pairs.append((int(addr, 16), int(value, 16)))
    return pairs


def build_cfg_image(pairs: list[tuple[int, int]]) -> bytes:
    image = bytearray(0xC8)
    for addr, value in pairs:
        if 0 <= addr < len(image):
            image[addr] = value & 0xFF
    return bytes(image)


def load_bias_le(path: Path) -> bytes:
    """bias.hex is one 32-bit INT32 per line (8 hex digits). The bias
    SRAM is byte-addressed little-endian (boot_pkg.sv: DSCNN_BIAS), so
    emit 4 LE bytes per word."""
    out = bytearray()
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            out += (int(text, 16) & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(out)


def do_boot(ser, logmel_dir: Path, weights_path: Path, bias_path: Path,
            cfg_path: Path, verbose: bool = True) -> bool:
    loads = [
        ("log_lut", make_target(MOD_FEATURES, FEAT_LOG_LUT), 0,
         pack_16bit_le(load_hex16(logmel_dir / "log2_lut.hex"))),
        ("mel_coeff", make_target(MOD_FEATURES, FEAT_MEL_COEFF), 0,
         pack_16bit_le(load_hex16(logmel_dir / "mel_coeffs_sparse.hex"))),
        ("mel_meta", make_target(MOD_FEATURES, FEAT_MEL_META), 0,
         bytes(load_hex8(logmel_dir / "mel_indices.hex"))),
        ("weights", make_target(MOD_DSCNN, DSCNN_WEIGHTS), 0,
         bytes(load_hex8(weights_path))),
        ("bias", make_target(MOD_DSCNN, DSCNN_BIAS), 0,
         load_bias_le(bias_path)),
        ("cfg", make_target(MOD_DSCNN, DSCNN_CFG), 0,
         build_cfg_image(load_cfg_pairs(cfg_path))),
        ("cfg_done", make_target(MOD_DSCNN, DSCNN_CFG), 0xFF, b"\x00"),
    ]

    for name, target, addr, payload in loads:
        resp = send_packet(ser, target, addr, payload, ack_timeout_s=3.0)
        if verbose:
            print(f"  {name:10s} {len(payload):5d} bytes -> {response_name(resp)}")
        if resp != ACK_BYTE:
            return False

    resp = send_packet(ser, make_target(MOD_CONTROL, CTRL_BOOT_DONE), 0, b"")
    if verbose:
        print(f"  boot_done   sent       -> {response_name(resp)}")
    return resp == ACK_BYTE


def wait_for_class(ser, settle_s: float = 0.30, collect_s: float = 2.0) -> int | None:
    ser.reset_input_buffer()
    time.sleep(settle_s)

    deadline = time.monotonic() + collect_s
    while time.monotonic() < deadline:
        for byte in ser.read(64):
            if (byte & CLASS_TAG_MASK) == CLASS_TAG:
                return byte & 0x07
    return None


def mode_listen(ser) -> int:
    print(f"listening on {ser.port} @ {ser.baudrate} baud")
    try:
        while True:
            chunk = ser.read(64)
            if not chunk:
                continue
            tagged = [b & 0x07 for b in chunk if (b & CLASS_TAG_MASK) == CLASS_TAG]
            suffix = f"  [class={tagged[-1]}]" if tagged else ""
            print(" ".join(f"{b:02X}" for b in chunk) + suffix)
    except KeyboardInterrupt:
        print()
    return 0


def mode_probe(ser) -> int:
    tests = [
        ("log_lut", make_target(MOD_FEATURES, FEAT_LOG_LUT), pack_16bit_le([0x1234])),
        ("mel_coeff", make_target(MOD_FEATURES, FEAT_MEL_COEFF), pack_16bit_le([0x5678])),
        ("weights", make_target(MOD_DSCNN, DSCNN_WEIGHTS), bytes([0xAB, 0xCD])),
        ("cfg", make_target(MOD_DSCNN, DSCNN_CFG), bytes([0x12])),
    ]

    failures = 0
    for name, target, payload in tests:
        resp = send_packet(ser, target, 0, payload, ack_timeout_s=0.5)
        failures += int(resp != ACK_BYTE)
        print(f"  {name:10s} -> {response_name(resp)}")
    print(f"\nprobe: {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


def mode_boot(ser, args) -> int:
    return 0 if do_boot(ser, args.logmel_dir, args.weights, args.bias, args.cfg) else 1


def run_inference(ser, samples) -> int:
    resp = send_packet(ser, make_target(MOD_CONTROL, CTRL_SESSION_RESET), 0, b"")
    if resp != ACK_BYTE:
        print(f"  session_reset -> {response_name(resp)}")
        return 1

    print(f"\nStreaming {len(samples)} samples ({len(samples) * 2} bytes)...")
    start = time.monotonic()
    acks = stream_audio(ser, samples)
    print(f"  {acks} chunk ACKs in {time.monotonic() - start:.2f} s")

    print("\nWaiting for inference result...")
    cls = wait_for_class(ser)
    if cls is None:
        print("  no class tag received")
        return 1
    print(f"  class_out = {cls}")
    return 0


def mode_stream(ser, args) -> int:
    print("Booting...")
    if not do_boot(ser, args.logmel_dir, args.weights, args.bias, args.cfg):
        return 1
    return run_inference(ser, sine_samples(SAMPLES_STREAM, 1000.0))


def mode_classify(ser, args) -> int:
    print("Booting...")
    if not do_boot(ser, args.logmel_dir, args.weights, args.bias, args.cfg):
        return 1
    return run_inference(ser, wav_samples(args.classify, n_required=SAMPLES_STREAM))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="/dev/ttyUSB1")
    parser.add_argument("-b", "--baud", type=int, default=460800)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--listen-only", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--boot", action="store_true")
    mode.add_argument("--stream", action="store_true")
    mode.add_argument("--classify", type=Path, metavar="WAV")

    parser.add_argument("--logmel-dir", type=Path, default=LOGMEL_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL_DIR / "weights.hex")
    parser.add_argument("--bias", type=Path, default=DEFAULT_MODEL_DIR / "bias.hex")
    parser.add_argument("--cfg", type=Path, default=FULL_DEMO_HOST_DIR / "cfg.hex")
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
        if args.listen_only:
            return mode_listen(ser)
        if args.probe:
            return mode_probe(ser)
        if args.boot:
            return mode_boot(ser, args)
        if args.stream:
            return mode_stream(ser, args)
        if args.classify:
            return mode_classify(ser, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
