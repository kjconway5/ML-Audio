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
    DBG_READ_SPECT_A,
    DBG_READ_SPECT_B,
    DEFAULT_MODEL_DIR,
    DSCNN_BIAS,
    DSCNN_CFG,
    DSCNN_WEIGHTS,
    ERR_BYTE,
    FEAT_LOG_LUT,
    FEAT_MEL_COEFF,
    FEAT_MEL_META,
    FULL_DEMO_HOST_DIR,
    LOGMEL_DIR,
    MOD_AUDIO,
    MOD_CONTROL,
    MOD_DEBUG,
    MOD_DSCNN,
    MOD_FEATURES,
    NACK_BYTE,
    SAMPLE_RATE,
    SAMPLES_STREAM,
    frame_packet,
    make_target,
    mic_samples,
    pack_16bit_le,
    response_name,
    send_packet,
    sine_samples,
    stream_audio,
    load_hex16,
    load_hex8,
    wav_samples,
)

# Class label order MUST match how the trained checkpoint was labeled,
# which is sorted() alphabetically (see run_rtl_wav.load_model:169).
# config.yaml's class list is in a DIFFERENT order — don't use it here.
CLASS_LABELS = ["no", "off", "on", "silence", "unknown", "wow", "yes"]


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


def read_spect_bank(ser, bank: str, byte_count: int = 2000,
                    ack_timeout_s: float = 1.0,
                    data_timeout_s: float = 5.0) -> bytes | None:
    """Walk one spectrogram-SRAM bank back over UART via MOD_DEBUG.
    bank is 'a' or 'b'; byte_count defaults to N_MELS(40)*N_FRAMES(50)=2000.

    Protocol: send DBG_READ_SPECT_{A,B} with len=byte_count and a
    matching dummy payload (boot_controller's S_RD_PAYLOAD requires
    `len` bytes). After it ACKs, spect_streamer streams `byte_count`
    int8 spectrogram bytes back on the same TX line.
    """
    sub = DBG_READ_SPECT_A if bank.lower() == 'a' else DBG_READ_SPECT_B
    target = make_target(MOD_DEBUG, sub)
    dummy = b"\x00" * byte_count

    # Aggressive drain: stream mode leaves stale class-tag bytes (0xC0)
    # in the UART pipeline that beat our ACK. reset_input_buffer alone
    # isn't enough — FT2232 USB endpoints can deliver in-flight bytes
    # AFTER the flush. Read+discard for a settling window, then flush
    # again, then send.
    ser.reset_input_buffer()
    drain_deadline = time.monotonic() + 0.15
    while time.monotonic() < drain_deadline:
        if not ser.read(64):
            break
    ser.reset_input_buffer()

    ser.write(frame_packet(target, 0, dummy))
    ser.flush()

    # 1) Wait for the boot_controller ACK. After boot_done, kws_top.start
    # is held high, so class_reporter keeps emitting class-tag bytes that
    # interleave on the shared TX. Skip those while waiting for ACK.
    deadline = time.monotonic() + ack_timeout_s
    ack = None
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if (b[0] & CLASS_TAG_MASK) == CLASS_TAG:
            continue  # in-band class byte from class_reporter — discard
        ack = b[0]
        break
    if ack != ACK_BYTE:
        print(f"  bank {bank}: no ACK (got {ack!r}) — check baud / boot state")
        return None

    # 2) Collect byte_count data bytes from spect_streamer
    received = bytearray()
    deadline = time.monotonic() + data_timeout_s
    while len(received) < byte_count and time.monotonic() < deadline:
        chunk = ser.read(byte_count - len(received))
        if chunk:
            received.extend(chunk)
    if len(received) != byte_count:
        print(f"  bank {bank}: short read {len(received)}/{byte_count}")
        return None
    return bytes(received)


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
    label = CLASS_LABELS[cls] if 0 <= cls < len(CLASS_LABELS) else "?"
    print(f"  class_out = {cls} ({label})")
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


def mode_mic_continuous(ser, args) -> int:
    """Continuous live-streaming mode — what the ASIC would naturally do:
    mic always on, audio flowing into pipeline_top, kws_top fires inference
    on every spect_done (~every 400 ms after the initial ramp), class bytes
    stream back. We don't wait between captures; we just keep the FPGA
    fed and print classes as they arrive.

    Threading:
      - sounddevice callback enqueues mic chunks (16-bit mono @ 16 kHz)
      - sender thread frames each chunk as a MOD_AUDIO packet → UART
      - reader thread reads UART continuously; class-tag bytes → printed,
        ACK bytes counted, NACK/ERR flagged
    """
    import queue
    import threading
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed; pip install sounddevice")
        return 1

    print("Booting...")
    if not do_boot(ser, args.logmel_dir, args.weights, args.bias, args.cfg):
        return 1

    # Clean start so the first reported class lines up with the first audio
    print("\nSession reset...")
    resp = send_packet(ser, make_target(MOD_CONTROL, CTRL_SESSION_RESET), 0, b"")
    if resp != ACK_BYTE:
        print(f"  session_reset → {response_name(resp)}")
        return 1
    ser.reset_input_buffer()

    BLOCKSIZE   = 2048               # samples per mic chunk → packet
    audio_q     = queue.Queue(maxsize=64)
    stop_event  = threading.Event()
    audio_target = make_target(MOD_AUDIO, 0)

    # --- Mic capture: callback runs on sounddevice's audio thread
    def mic_cb(indata, frames, time_info, status):
        if status:
            print(f"  [mic] {status}")
        try:
            audio_q.put_nowait(indata[:, 0].astype("<i2").tobytes())
        except queue.Full:
            # Drop chunks if sender can't keep up rather than block the
            # audio thread (would cause underruns / glitchy capture).
            print("  [mic] queue full — dropped chunk")

    # --- Sender: take chunks from queue, frame as packets, write to UART
    def sender_loop():
        sent_chunks = 0
        while not stop_event.is_set():
            try:
                payload = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            ser.write(frame_packet(audio_target, 0, payload))
            sent_chunks += 1

    # --- Reader: stream UART, surface classes, count ACKs, flag errors.
    # kws_top re-runs inference at ~20 Hz on the same spect bank between
    # spect_done pulses, producing long runs of identical class outputs.
    # Default view dedupes those runs (one line per CHANGE, with the run
    # length); --mic-verbose dumps every class byte for forensics.
    def reader_loop():
        ack_count = 0
        cls_count = 0
        nack_count = 0
        last_cls = None
        run_len  = 0
        run_started = time.monotonic()

        def flush_run():
            nonlocal last_cls, run_len, run_started
            if last_cls is None:
                return
            label = CLASS_LABELS[last_cls] if 0 <= last_cls < len(CLASS_LABELS) else "?"
            held = time.monotonic() - run_started
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}]  class={last_cls}  {label:<8s}"
                  f"  ×{run_len:<3d}  ({held*1000:.0f} ms)"
                  f"   acks={ack_count} nacks={nack_count} total_cls={cls_count}")

        while not stop_event.is_set():
            data = ser.read(64)
            if not data:
                # No bytes for a while: flush a stale run so the user sees
                # something even during silence.
                if last_cls is not None and (time.monotonic() - run_started) > 0.5:
                    flush_run()
                    last_cls = None; run_len = 0
                continue
            for b in data:
                if (b & CLASS_TAG_MASK) == CLASS_TAG:
                    cls_count += 1
                    cls = b & 0x07
                    if args.mic_verbose:
                        label = CLASS_LABELS[cls] if 0 <= cls < len(CLASS_LABELS) else "?"
                        ts = time.strftime("%H:%M:%S")
                        print(f"  [{ts}]  #{cls_count:4d}  class={cls}  {label}")
                    else:
                        if cls == last_cls:
                            run_len += 1
                        else:
                            flush_run()
                            last_cls = cls
                            run_len  = 1
                            run_started = time.monotonic()
                elif b == ACK_BYTE:
                    ack_count += 1
                elif b == NACK_BYTE:
                    nack_count += 1
                elif b == ERR_BYTE:
                    print(f"  [uart] ERR byte (malformed packet) — host/fpga lost sync?")
        flush_run()  # final flush on shutdown

    sender_t = threading.Thread(target=sender_loop, daemon=True)
    reader_t = threading.Thread(target=reader_loop, daemon=True)
    sender_t.start()
    reader_t.start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCKSIZE,
        callback=mic_cb,
        device=args.mic_device,
    )

    print(f"\nLive-stream mode — say keywords ({CLASS_LABELS}) anytime.")
    print(f"  blocksize={BLOCKSIZE} samples ({BLOCKSIZE/SAMPLE_RATE*1000:.0f} ms/chunk)")
    print(f"  expected class rate ≈ 2.5 Hz (one per ~400 ms of audio after ramp-up)")
    print(f"  Ctrl-C to stop.\n")
    try:
        with stream:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n  stopping...")
    finally:
        stop_event.set()
        sender_t.join(timeout=1)
        reader_t.join(timeout=1)
    return 0


def mode_mic(ser, args) -> int:
    """One-shot mic mode: ENTER → record N seconds → infer → print.

    For the chip-like continuous mode (mic always on, classes always
    coming out), use --mic-continuous instead.

    Boot once, then in a loop: record 1 s from the system mic, run
    one FPGA inference, print the class. Ctrl-C to stop.
    Tip: speak a keyword roughly centered in the 1 s recording window."""
    print("Booting...")
    if not do_boot(ser, args.logmel_dir, args.weights, args.bias, args.cfg):
        return 1

    print(f"\nLive mic mode — say one of: {CLASS_LABELS}")
    interactive = (args.mic_loops is None)
    if interactive:
        print(f"  (records {args.mic_seconds:.2f}s, infers, prints class. Ctrl-C to stop.)\n")
    else:
        print(f"  (running {args.mic_loops} back-to-back captures of {args.mic_seconds:.2f}s each)\n")
    try:
        i = 0
        while True:
            i += 1
            if interactive:
                try:
                    input(f"  [{i:3d}] press ENTER to record...")
                except EOFError:
                    return 0
            else:
                if i > args.mic_loops:
                    return 0
                print(f"  [{i:3d}/{args.mic_loops}] recording {args.mic_seconds:.2f}s now —")
            samples = mic_samples(seconds=args.mic_seconds,
                                  n_required=SAMPLES_STREAM,
                                  device=args.mic_device)
            peak = int(abs(samples).max())
            print(f"        recorded peak={peak} ({peak/32768*100:.1f}% full-scale)")
            run_inference(ser, samples)
    except KeyboardInterrupt:
        print("\n  stopped.")
        return 0


def mode_read_spect(ser, args) -> int:
    """Dump one or both spectrogram banks (int8) to stdout / file."""
    banks = ['a', 'b'] if args.read_spect == 'both' else [args.read_spect]
    failures = 0
    for b in banks:
        data = read_spect_bank(ser, b)
        if data is None:
            failures += 1
            continue
        # Reshape 2000 int8 bytes → 50 frames × 40 mels (signed)
        import numpy as np  # local import: only this mode needs it
        arr = np.frombuffer(data, dtype=np.int8).reshape(50, 40)
        out = args.spect_out / f"spect_bank_{b}.npy" if args.spect_out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, arr)
            print(f"  bank {b}: {len(data)} bytes → {out} (shape {arr.shape}, "
                  f"min={arr.min()} max={arr.max()})")
        else:
            print(f"  bank {b}: {len(data)} bytes, shape {arr.shape}, "
                  f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
    return 1 if failures else 0


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
    mode.add_argument("--read-spect", choices=['a', 'b', 'both'],
                      help="dump spectrogram bank(s) via MOD_DEBUG (run after streaming audio)")
    mode.add_argument("--mic", action="store_true",
                      help="one-shot mic mode: boot once, ENTER to record+classify")
    mode.add_argument("--mic-continuous", action="store_true",
                      help="chip-like streaming: mic always on, classes always coming out")

    parser.add_argument("--logmel-dir", type=Path, default=LOGMEL_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL_DIR / "weights.hex")
    parser.add_argument("--bias", type=Path, default=DEFAULT_MODEL_DIR / "bias.hex")
    parser.add_argument("--cfg", type=Path, default=FULL_DEMO_HOST_DIR / "cfg.hex")
    parser.add_argument("--spect-out", type=Path, default=None,
                        help="directory for --read-spect .npy dumps (omit = print stats only)")
    parser.add_argument("--mic-seconds", type=float, default=1.0,
                        help="seconds to record per --mic iteration (default 1.0)")
    parser.add_argument("--mic-device", default=None,
                        help="sounddevice input device index or name (default = system default)")
    parser.add_argument("--mic-loops", type=int, default=None,
                        help="non-interactive: record N times back-to-back (default = interactive ENTER prompt)")
    parser.add_argument("--mic-verbose", action="store_true",
                        help="--mic-continuous: print every class byte (default dedupes runs)")
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
        if args.read_spect:
            return mode_read_spect(ser, args)
        if args.mic:
            return mode_mic(ser, args)
        if args.mic_continuous:
            return mode_mic_continuous(ser, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
