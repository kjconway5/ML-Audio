#!/usr/bin/env python3
"""
Demo2: Host-Mic → FPGA → Host Inference

Records audio from the computer's microphone, streams 14-bit samples
to the FPGA over UART, receives log-mel feature packets back, and
runs the DSCNN model for keyword spotting.

Usage:
    python stream_demo2.py -p /dev/ttyUSB1 -m tiny-7class-golden.pt
    python stream_demo2.py -p /dev/ttyUSB1 -m tiny-7class-golden.pt --list-devices
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DSCNN

# Protocol constants
VERSION = 0x01
N_MELS = 40
HEADER_LEN = 6
PAYLOAD_LEN = N_MELS * 2
PACKET_LEN = HEADER_LEN + PAYLOAD_LEN + 1  # 87
Q_FRAC = 12
SAMPLE_RATE = 16000
SAMPLE_MAX = (1 << 13) - 1  # 8191, 14-bit signed
N_FRAMES = 124
INFER_EVERY = 31  # ~250 ms
BAUD = 460800


# ------------------------------------------------------------------ #
#  Mic capture                                                         #
# ------------------------------------------------------------------ #
def start_mic_sender(ser, device=None):
    """Start a daemon thread that records audio and streams 14-bit samples over UART."""

    def callback(indata, frames, time_info, status):
        samples = indata[:, 0]
        scaled = np.clip(samples.astype(np.float32) / 4.0, -SAMPLE_MAX, SAMPLE_MAX).astype(np.int16)
        ser.write(scaled.tobytes())

    def run():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=512, device=device, callback=callback):
            threading.Event().wait()

    threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------ #
#  Packet reader                                                       #
# ------------------------------------------------------------------ #
def read_packet(ser):
    """Block until a valid feature packet arrives. Returns (frame_id, features_u16) or None."""
    while True:
        b = ser.read(1)
        # nothing read, return None
        if len(b) == 0:
            return None
        # first byte of packet start signal found, check next
        if b[0] == 0xAA:
            b2 = ser.read(1)
            if len(b2) == 0:
                return None
            # valid packet found (starts w/ 0xAA, 0x55) so break
            if b2[0] == 0x55:
                break

    # read the last 85 bytes for the full packet 
    rest = ser.read(PACKET_LEN - 2)
    if len(rest) < PACKET_LEN - 2:
        return None

    # full assembled packet
    raw = bytes([0xAA, 0x55]) + rest

    # xor first 86 bytes, compare against 87th byte to see if checksum matches
    chk = 0
    for b in raw[: PACKET_LEN - 1]:
        chk ^= b
    if chk != raw[PACKET_LEN - 1]:
        return None

    # double check VERSION and N_MELS, return None if they arent correct
    if raw[2] != VERSION or raw[5] != N_MELS:
        return None

    # reconstruct 16 bit int
    frame_id = (raw[3] << 8) | raw[4]
    # take last 80 bytes of packet which are the features, 2 bytes per mel bin
    features = np.frombuffer(raw[HEADER_LEN : HEADER_LEN + PAYLOAD_LEN], dtype=np.uint16).copy()
    return frame_id, features


# ------------------------------------------------------------------ #
#  Frame buffer                                                        #
# ------------------------------------------------------------------ #
def make_frame_buffer():
    """Return (push_fn, ready_fn, snapshot_fn) over a rolling buffer."""
    # create 124 x 40 zeroed matrix (frames x n_mels)
    buf = np.zeros((N_FRAMES, N_MELS), dtype=np.float32)
    fill = [0]

    # pop oldest frame, append newest to that space
    def push(features_u16):
        buf[:-1] = buf[1:]
        buf[-1] = features_u16.astype(np.float32) / (1 << Q_FRAC)
        # tracks how many frames have arrived
        fill[0] = min(fill[0] + 1, N_FRAMES)

    # returns push function, check for 124 frames arrived,
    # copy of current buffer, and the fill counter
    return push, lambda: fill[0] >= N_FRAMES, lambda: buf.copy(), fill


# ------------------------------------------------------------------ #
#  Model loader + inference                                            #
# ------------------------------------------------------------------ #
def load_model(path):
    # load model checkpoint
    cp = torch.load(path, map_location="cpu", weights_only=False)
    labels = cp["labels"]

    model = DSCNN(n_classes=len(labels))

    # QAT reconstruction (demo_one.pt is quantized)
    model.eval(); model.fuse_model(); model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig(cp["qat_backend"])
    torch.quantization.prepare_qat(model, inplace=True)
    model.eval()
    torch.quantization.convert(model, inplace=True)

    # load weights after QAT reconstruction
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, labels


def classify(model, snapshot, labels):
    x = torch.from_numpy(snapshot).float()
    x = x - x.mean()
    std = x.std()
    if std > 1e-6:
        x = x / std
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0).unsqueeze(0)), dim=1)
    idx = probs.argmax(dim=1).item()
    return labels[idx], probs[0, idx].item()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Demo: mic → FPGA features → KWS inference")
    parser.add_argument("-p", "--port", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Loading model: {args.model}")
    model, labels = load_model(args.model)
    print(f"  Classes: {labels}")

    print(f"Opening serial: {args.port} @ {BAUD}")
    ser = serial.Serial(args.port, BAUD, timeout=0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.1)

    push, ready, snapshot, fill = make_frame_buffer()
    start_mic_sender(ser, device=args.device)

    frames_since_infer = 0
    t_start = time.time()

    print("Starting mic capture + FPGA streaming...\n")

    try:
        while True:
            result = read_packet(ser)
            if result is None:
                continue

            frame_id, features = result
            push(features)
            frames_since_infer += 1

            if not ready():
                print(f"\r  Buffering: {fill[0]}/{N_FRAMES} frames", end="", flush=True)
                continue

            if frames_since_infer >= INFER_EVERY:
                frames_since_infer = 0
                label, conf = classify(model, snapshot(), labels)
                elapsed = time.time() - t_start

                print(f"\r[{elapsed:6.1f}s] frame={frame_id:5d}  "
                      f"pred={label:>10s} {conf*100:5.1f}%    ", flush=True)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
