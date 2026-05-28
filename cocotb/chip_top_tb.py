# SPDX-FileCopyrightText: © 2025 Project Template Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import time
import logging
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, RisingEdge
from cocotb_tools.runner import get_runner

sim = os.getenv("SIM", "icarus")
pdk_root = os.getenv("PDK_ROOT", Path("~/.ciel").expanduser())
pdk = os.getenv("PDK", "gf180mcuD")
scl = os.getenv("SCL", "gf180mcu_fd_sc_mcu7t5v0")
gl = os.getenv("GL", False)
slot = os.getenv("SLOT", "1x1")

hdl_toplevel = "chip_top"

async def set_defaults(dut):
    dut.input_PAD.value = 0
    # UART RX is bidir_PAD[0]. Hold it idle-high and release every other
    # bidirectional pad so chip_top can drive its output pads without contention.
    dut.bidir_PAD.value = "Z" * 39 + "1"

async def enable_power(dut):
    dut.VDD.value = 1
    dut.VSS.value = 0

async def start_clock(clock, freq=25):
    """Start the clock @ freq MHz"""
    c = Clock(clock, 1 / freq * 1000, "ns")
    cocotb.start_soon(c.start())


async def reset(reset, active_low=True, time_ns=1000):
    """Reset dut"""
    cocotb.log.info("Reset asserted...")

    reset.value = not active_low
    await Timer(time_ns, "ns")
    reset.value = active_low

    cocotb.log.info("Reset deasserted.")


async def start_up(dut):
    """Startup sequence"""
    await set_defaults(dut)
    if gl:
        await enable_power(dut)
    await start_clock(dut.clk_PAD)
    await reset(dut.rst_n_PAD)


def _pad_bit(value, bit_index):
    # str(LogicArray) is MSB-first, while pad bit numbers are LSB-first.
    return str(value)[-(bit_index + 1)].lower()



def _pad_bit_value(value, bit_index, default=0):
    ch = _pad_bit(value, bit_index)
    if ch == "1":
        return 1
    if ch == "0":
        return 0
    return default


# These constants mirror chip_core_tb.py and chip_core.sv. The top-level test
# reaches them through pads instead of direct chip_core ports.
NUM_INPUT_PADS = 12
NUM_BIDIR_PADS = 40
UART_RX_PAD = 0
UART_TX_PAD = 1
KWS_DONE_PAD = 2
KWS_CLASS_BASE = 3
CLK_PERIOD_NS = 40
UART_PRESCALE = 1
BIT_CYCLES = UART_PRESCALE * 8
SYNC_0 = 0xAA
SYNC_1 = 0x55
ACK_BYTE = 0x06
MOD_FEATURES = 0x0
MOD_DSCNN = 0x1
MOD_CONTROL = 0xF
FEAT_LOG_LUT = 0x0
FEAT_MEL_COEFF = 0x1
FEAT_MEL_META = 0x2
DSCNN_WEIGHTS = 0x0
DSCNN_CFG = 0x1
DSCNN_BIAS = 0x2
CTRL_BOOT_DONE = 0x0
MODEL_FILTERS = 16

_SRC = Path(__file__).resolve().parent.parent / "src"
_RTL = _SRC / "rtl"
_ML = _SRC / "ml"
_KWS_DIR = _RTL / "dscnn/kws_top"
_LOGMEL = _RTL / "Log-Mel/data"
_MODEL_DIR = _ML / "models/dscnn-16center-v1"
LOG_LUT_HEX = _LOGMEL / "log2_lut.hex"
MEL_COEFF_HEX = _LOGMEL / "mel_coeffs_sparse.hex"
MEL_INDEX_HEX = _LOGMEL / "mel_indices.hex"
WEIGHTS_HEX = _MODEL_DIR / "weights.hex"
BIAS_HEX = _MODEL_DIR / "bias.hex"
SCALES_TXT = _MODEL_DIR / "scales.txt"
SPECT_DIR = _KWS_DIR / "spectrograms"
MANIFEST_JSON = SPECT_DIR / "test_vectors.json"
RESULTS_TXT = _MODEL_DIR / "sim-top-results.txt"

sys.path.insert(0, str(_KWS_DIR))
from rtl_golden import load_layer_cfgs  # noqa: E402


def _make_target(mod, sub):
    return ((mod & 0xF) << 4) | (sub & 0xF)


def _manifest_json_path():
    manifest_env = os.getenv("KWS_MANIFEST_JSON")
    if not manifest_env:
        return MANIFEST_JSON

    path = Path(manifest_env)
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd() / path, repo_root / path, _KWS_DIR / path):
        if candidate.exists():
            return candidate
    return repo_root / path


def _load_hex(path, signed=False, width=8):
    vals = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            value = int(text, 16)
            if signed and value >= (1 << (width - 1)):
                value -= (1 << width)
            vals.append(value)
    return vals


def _pack_layer_cfgs(layer_cfgs):
    buf = [0] * 0xC8
    for (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
         dw, w_off, mult, shift, relu, ofmap_h, ofmap_w, bias_off) in layer_cfgs:
        base = layer << 4
        buf[base | 0x0] = in_ch & 0xFF
        buf[base | 0x1] = out_ch & 0xFF
        buf[base | 0x2] = kH & 0xFF
        buf[base | 0x3] = kW & 0xFF
        buf[base | 0x4] = sh & 0xFF
        buf[base | 0x5] = sw & 0xFF
        buf[base | 0x6] = ph & 0xFF
        buf[base | 0x7] = pw & 0xFF
        buf[base | 0x8] = dw & 0xFF
        buf[base | 0x9] = w_off & 0xFF
        buf[base | 0xA] = (w_off >> 8) & 0x1F
        buf[base | 0xB] = shift & 0xFF
        buf[base | 0xC] = (relu & 0x1) | ((bias_off >> 7) & 0x2)
        buf[base | 0xD] = ofmap_h & 0xFF
        buf[base | 0xE] = ofmap_w & 0xFF
        buf[base | 0xF] = bias_off & 0xFF

        mult_base = 0xA0 + layer * 4
        buf[mult_base + 0] = (mult >> 0) & 0xFF
        buf[mult_base + 1] = (mult >> 8) & 0xFF
        buf[mult_base + 2] = (mult >> 16) & 0xFF
        buf[mult_base + 3] = (mult >> 24) & 0xFF
    return buf[0x00:0xA0], buf[0xA0:0xC8]


def _build_packet(target, addr, payload):
    payload = list(payload)
    body = [target, (addr >> 8) & 0xFF, addr & 0xFF, (len(payload) >> 8) & 0xFF, len(payload) & 0xFF] + payload
    checksum = 0
    for byte in body:
        checksum ^= byte
    return [SYNC_0, SYNC_1] + body + [checksum & 0xFF]


def _drive_bidir_uart_rx(dut, bit):
    # Only external UART RX, bidir_PAD[0], is driven by the testbench. All other
    # bidirectional pads remain Z so chip_top can drive TX/KWS outputs.
    dut.bidir_PAD.value = "Z" * (NUM_BIDIR_PADS - 1) + ("1" if bit else "0")


def _uart_tx_pad_bit(dut):
    return _pad_bit_value(dut.bidir_PAD.value, UART_TX_PAD, default=1)


async def _uart_drive_byte(dut, byte_val, log=None):
    if log:
        log.debug(f"  UART RX drive byte 0x{byte_val:02X}")
    _drive_bidir_uart_rx(dut, 0)
    await ClockCycles(dut.clk_PAD, BIT_CYCLES)
    for i in range(8):
        _drive_bidir_uart_rx(dut, (byte_val >> i) & 1)
        await ClockCycles(dut.clk_PAD, BIT_CYCLES)
    _drive_bidir_uart_rx(dut, 1)
    await ClockCycles(dut.clk_PAD, BIT_CYCLES)


async def _uart_read_byte(dut, timeout_cycles=800_000):
    log = dut._log
    log.info("  [UART RX] waiting for start bit on bidir_PAD[1] (chip TX)")
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk_PAD)
        if _uart_tx_pad_bit(dut) == 0:
            log.info(f"  [UART RX] start bit detected at poll cycle {cyc}")
            break
        if cyc % 50000 == 49999:
            log.info(f"  [UART RX] still idle at poll cycle {cyc}, TX={_uart_tx_pad_bit(dut)}")
    else:
        raise AssertionError("UART RX: timeout waiting for chip TX start bit on bidir_PAD[1]")

    await ClockCycles(dut.clk_PAD, BIT_CYCLES + BIT_CYCLES // 2)
    byte_val = 0
    for i in range(8):
        byte_val |= (_uart_tx_pad_bit(dut) << i)
        if i < 7:
            await ClockCycles(dut.clk_PAD, BIT_CYCLES)
    await ClockCycles(dut.clk_PAD, BIT_CYCLES)
    log.info(f"  [UART RX] received byte 0x{byte_val:02X}")
    return byte_val


async def _send_packet(dut, target, addr, payload):
    log = dut._log
    pkt = _build_packet(target, addr, payload)
    log.info(f"  [PKT] target=0x{target:02X} addr=0x{addr:04X} payload_len={len(payload)} total_bytes={len(pkt)}")
    for idx, byte in enumerate(pkt):
        if idx < 7 or idx == len(pkt) - 1:
            log.info(f"    TX[{idx}] = 0x{byte:02X}")
        elif idx == 7:
            log.info(f"    TX[7..{len(pkt)-2}] = payload ({len(payload)} bytes) ...")
        await _uart_drive_byte(dut, byte, log=log)
    log.info("  [PKT] all bytes sent, awaiting ACK ...")
    resp = await _uart_read_byte(dut)
    assert resp == ACK_BYTE, f"Expected ACK 0x{ACK_BYTE:02X} for target=0x{target:02X} addr=0x{addr:04X}, got 0x{resp:02X}"
    log.info("  [PKT] ACK received OK")


async def _boot_chip_from_pads(dut):
    lut_words = _load_hex(LOG_LUT_HEX, signed=False, width=16)
    mel_words = _load_hex(MEL_COEFF_HEX, signed=False, width=16)
    meta_bytes = _load_hex(MEL_INDEX_HEX, signed=False, width=8)
    weight_bytes = _load_hex(WEIGHTS_HEX, signed=True, width=8)
    bias_words = _load_hex(BIAS_HEX, signed=True, width=32)
    cfg_fields, cfg_mults = _pack_layer_cfgs(load_layer_cfgs(SCALES_TXT, n_filters=MODEL_FILTERS))

    lut_payload = []
    for word in lut_words:
        lut_payload += [word & 0xFF, (word >> 8) & 0xFF]
    await _send_packet(dut, _make_target(MOD_FEATURES, FEAT_LOG_LUT), 0, lut_payload)
    dut._log.info(f"  Log LUT loaded ({len(lut_words)} words)")

    mel_payload = []
    for word in mel_words:
        mel_payload += [word & 0xFF, (word >> 8) & 0xFF]
    await _send_packet(dut, _make_target(MOD_FEATURES, FEAT_MEL_COEFF), 0, mel_payload)
    dut._log.info(f"  Mel coeffs loaded ({len(mel_words)} words)")

    await _send_packet(dut, _make_target(MOD_FEATURES, FEAT_MEL_META), 0, meta_bytes)
    dut._log.info(f"  Mel indices loaded ({len(meta_bytes)} bytes)")

    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_WEIGHTS), 0, [b & 0xFF for b in weight_bytes])
    dut._log.info(f"  Weights loaded ({len(weight_bytes)} bytes)")

    bias_payload = []
    for bias in bias_words:
        raw = bias & 0xFFFFFFFF
        bias_payload += [raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF, (raw >> 24) & 0xFF]
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_BIAS), 0, bias_payload)
    dut._log.info(f"  Biases loaded ({len(bias_words)} INT32 values)")

    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_CFG), 0x00, cfg_fields)
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_CFG), 0xA0, cfg_mults)
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_CFG), 0xFF, [0x01])
    dut._log.info("  Layer configs loaded, cfg_load_done set")

    await _send_packet(dut, _make_target(MOD_CONTROL, CTRL_BOOT_DONE), 0, [])
    dut._log.info("  boot_done asserted through pad-level UART")
    await ClockCycles(dut.clk_PAD, 10)


def _resolve_wav_path(wav_path):
    path = Path(wav_path)
    if path.exists():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    parts = path.parts
    if "src" in parts:
        src_index = parts.index("src")
        candidate = repo_root.joinpath(*parts[src_index:])
        if candidate.exists():
            return candidate

    candidate = repo_root / path
    if candidate.exists():
        return candidate

    return path


def _load_wav_pcm(wav_path, target_samples=16_000):
    import numpy as np
    wav_path = _resolve_wav_path(wav_path)
    sample_max = (1 << 13) - 1

    def _load_wav_pcm_stdlib(path):
        import wave
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sample_width == 1:
            data = np.frombuffer(frames, dtype=np.uint8)
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype="<i2")
        elif sample_width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            sign = (raw[:, 2] & 0x80) != 0
            pad = np.where(sign, 0xFF, 0x00).astype(np.uint8)
            data = np.column_stack((raw, pad)).reshape(-1).view("<i4")
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype="<i4")
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
        if channels > 1:
            data = data.reshape(-1, channels)
        return rate, data

    try:
        from scipy.io import wavfile
        rate, data = wavfile.read(str(wav_path))
    except Exception:
        try:
            import soundfile as sf
            data, rate = sf.read(str(wav_path))
        except Exception:
            rate, data = _load_wav_pcm_stdlib(wav_path)

    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.floating):
        audio = data.astype("float32")
    elif np.issubdtype(data.dtype, np.signedinteger):
        info = np.iinfo(data.dtype)
        audio = data.astype("float32") / float(abs(info.min))
    elif np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        midpoint = (info.max + 1) / 2.0
        audio = (data.astype("float32") - midpoint) / midpoint
    else:
        audio = data.astype("float32")

    if rate != 16_000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16_000, int(rate))
        audio = resample_poly(audio, 16_000 // g, rate // g).astype("float32")

    pcm = (np.clip(audio, -1.0, 1.0) * sample_max).astype("int32")
    if len(pcm) >= target_samples:
        return pcm[:target_samples]
    return np.concatenate([pcm, np.zeros(target_samples - len(pcm), dtype="int32")])


def _pcm_to_pdm(pcm, decim=63):
    pdm = []
    acc = 0
    for sample in pcm:
        for _ in range(decim):
            acc += int(sample)
            if acc >= 0:
                pdm.append(1)
                acc -= (1 << 15)
            else:
                pdm.append(0)
                acc += (1 << 15)
    return pdm


def _select_manifest_sample(samples):
    if not samples:
        raise AssertionError("test_vectors.json contains no samples")
    sample_keyword = os.getenv("KWS_KEYWORD")
    sample_index = os.getenv("KWS_SAMPLE_INDEX")
    sample_match = os.getenv("KWS_SAMPLE_MATCH")

    filtered = list(enumerate(samples))
    if sample_keyword:
        keyword = sample_keyword.lower()
        filtered = [(idx, sample) for idx, sample in filtered if str(sample.get("ground_truth_name", "")).lower() == keyword]
        if not filtered:
            available = sorted({str(s.get("ground_truth_name", "?")) for s in samples})
            raise AssertionError(f"KWS_KEYWORD={sample_keyword!r} did not match this manifest. Available ground_truth_name values: {available}.")

    if sample_index is not None:
        idx = int(sample_index, 0)
        if idx < 0 or idx >= len(samples):
            raise AssertionError(f"KWS_SAMPLE_INDEX={idx} is out of range; manifest has {len(samples)} samples")
        return idx, samples[idx]

    if sample_match:
        needle = sample_match.lower()
        for idx, sample in filtered:
            if needle in str(sample.get("wav", "")).lower() or needle in str(sample.get("ground_truth_name", "")).lower():
                return idx, sample
        raise AssertionError(f"KWS_SAMPLE_MATCH={sample_match!r} did not match any selected manifest sample")

    return filtered[0]


def _core(dut):
    return dut.i_chip_core


def _get_path(root, path):
    obj = root
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


def _handle_int(handle, default='?'):
    if handle is None:
        return default
    try:
        value = handle.value
        if hasattr(value, 'is_resolvable') and not value.is_resolvable:
            return 'X'
        return int(value)
    except Exception:
        return default


def _handle_is_one(handle):
    return _handle_int(handle, default=0) == 1


def _handle_signed_int(handle, width=32, default='?'):
    raw = _handle_int(handle, default=default)
    if raw in ('?', 'X'):
        return raw
    if raw >= (1 << (width - 1)):
        raw -= (1 << width)
    return raw


def _resolve_probe_handles(dut):
    paths = {
        'cic_valid': 'pipeline_inst.cic_valid',
        'fir_valid': 'pipeline_inst.fir_valid_o',
        'fft_sync': 'pipeline_inst.u_stfft.o_fft_sync',
        'fft_sync_aligned': 'pipeline_inst.fft_sync_aligned',
        'fft_valid': 'pipeline_inst.fft_valid',
        'power_valid': 'pipeline_inst.u_logmel.power_valid',
        'filterbank_done': 'pipeline_inst.u_logmel.filterbank_done',
        'log_done': 'pipeline_inst.u_logmel.log_done',
        'mel_valid': 'pipeline_inst.mel_valid',
        'spect_done': 'spect_done',
        'kws_start': 'kws_start',
        'kws_done': 'kws_done',
        'mel_data': 'pipeline_inst.mel_compensated',
        'spect_wr_addr': 'pipeline_inst.u_spect_buf.wr_addr',
    }
    handles = {}
    root = _core(dut)
    for name, path in paths.items():
        try:
            handles[name] = _get_path(root, path)
        except Exception:
            handles[name] = None
    return handles


def _probe_str(dut):
    handles = _resolve_probe_handles(dut)
    return (
        f"cic={_handle_int(handles['cic_valid'])} fir={_handle_int(handles['fir_valid'])} "
        f"fft_sync={_handle_int(handles['fft_sync'])}/{_handle_int(handles['fft_sync_aligned'])} "
        f"fft_valid={_handle_int(handles['fft_valid'])} pwr={_handle_int(handles['power_valid'])} "
        f"fb_done={_handle_int(handles['filterbank_done'])} log_done={_handle_int(handles['log_done'])} "
        f"mel={_handle_int(handles['mel_valid'])} spect_done={_handle_int(handles['spect_done'])} "
        f"kws_start={_handle_int(handles['kws_start'])} kws_done={_handle_int(handles['kws_done'])} "
        f"wr={_handle_int(handles['spect_wr_addr'])}"
    )


def _read_kws_pads(dut):
    value = dut.bidir_PAD.value
    done = _pad_bit_value(value, KWS_DONE_PAD, default=0)
    cls = ((_pad_bit_value(value, KWS_CLASS_BASE + 2, default=0) << 2)
           | (_pad_bit_value(value, KWS_CLASS_BASE + 1, default=0) << 1)
           | _pad_bit_value(value, KWS_CLASS_BASE, default=0))
    return done, cls


def _read_kws_scores(dut, class_names):
    scores = []
    try:
        root = _core(dut).kws_inst
    except Exception:
        return None, "kws_inst hierarchy is unavailable"
    for idx, name in enumerate(class_names[:7]):
        try:
            handle = getattr(root, f"debug_gap{idx}")
        except Exception:
            return None, "debug_gap score wires are unavailable"
        scores.append((idx, name, _handle_signed_int(handle, width=32)))
    if any(score in ('?', 'X') for _, _, score in scores):
        return scores, "one or more debug_gap scores are X/unavailable"
    return scores, None


def _format_kws_scores(scores):
    if not scores:
        return "scores unavailable"
    return "  ".join(f"{name}({idx})={score}" for idx, name, score in scores)


def _format_kws_score_ranking(scores):
    if not scores or any(score in ('?', 'X') for _, _, score in scores):
        return "ranking unavailable"
    ranked = sorted(scores, key=lambda item: item[2], reverse=True)
    margin = ranked[0][2] - ranked[1][2] if len(ranked) > 1 else 0
    return " > ".join(f"{name}({idx})={score}" for idx, name, score in ranked) + f"  margin={margin}"


def _class_label(name, cls):
    if name is None and cls is None:
        return "unavailable"
    return f"{name if name is not None else '?'} ({cls if cls is not None else '?'})"


async def _drive_pdm_from_pads(dut, pdm_bits):
    for bit in pdm_bits:
        dut.input_PAD.value = 0b10 | (bit & 1)
        await RisingEdge(dut.clk_PAD)
    dut.input_PAD.value = 0


async def _wait_for_kws_done_pad(dut, timeout_cycles):
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk_PAD)
        done, cls = _read_kws_pads(dut)
        if done:
            return cyc + 1, cls
    return None, None


def _append_sim_top_result(**kwargs):
    status = "PASS" if kwargs['passed'] else "FAIL"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"[{timestamp}] make sim chip_top",
        f"Model: {_MODEL_DIR.name}",
        f"Manifest: {kwargs['manifest_path']}",
        f"Selected sample[{kwargs['sample_idx']}]: {Path(kwargs['wav_path']).name}",
        f"WAV: {kwargs['wav_path']}",
        f"Hex: {kwargs['hex_file']}",
        f"Golden truth class: {_class_label(kwargs['gt_name'], kwargs['gt_class'])}",
        f"Arithmetic class: {_class_label(kwargs['arith_name'], kwargs['arith_class'])}",
        f"PyTorch class: {_class_label(kwargs['pytorch_name'], kwargs['pytorch_class'])}",
        f"RTL class from chip_top pads: {_class_label(kwargs['rtl_name'], kwargs['rtl_class'])}",
        f"Expected[{kwargs['expected_source']}]: {_class_label(kwargs['expected_name'], kwargs['expected_class'])}",
        f"Result: {status}",
        f"RTL GAP scores: {_format_kws_scores(kwargs['scores'])}",
        f"RTL GAP ranking: {_format_kws_score_ranking(kwargs['scores'])}",
    ]
    if kwargs['score_warning']:
        lines.append(f"RTL GAP score warning: {kwargs['score_warning']}")
    RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_TXT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")


@cocotb.test()
async def test_chip_top_e2e(dut):
    """
    End-to-end chip_top RTL verification through real top-level pads:
      1. Boot log/mel/KWS memories through bidir_PAD[0]/[1] UART.
      2. Drive PDM audio on input_PAD[1:0].
      3. Wait for KWS_DONE on bidir_PAD[2].
      4. Read KWS_CLASS from bidir_PAD[5:3] and compare against manifest output.
    """
    manifest_path = _manifest_json_path()
    for path in [LOG_LUT_HEX, MEL_COEFF_HEX, MEL_INDEX_HEX, WEIGHTS_HEX, BIAS_HEX, SCALES_TXT, manifest_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing: {path}\nRun src/ml/Pipeline/export.py and generate_spect_full.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)
    class_names = manifest["class_names"]
    samples = manifest["samples"]

    dut._log.info(f"Manifest: {manifest_path} ({len(samples)} samples)")
    await start_up(dut)

    dut._log.info("=== chip_top pad UART boot phase ===")
    await _boot_chip_from_pads(dut)

    sample_idx, sample = _select_manifest_sample(samples)
    gt_class = sample["ground_truth_class"]
    gt_name = sample["ground_truth_name"]
    arith_class = sample.get("arith_class")
    arith_name = sample.get("arith_name")
    pytorch_class = sample.get("pytorch_class")
    pytorch_name = sample.get("pytorch_name")
    wav_path = sample["wav"]
    resolved_wav_path = _resolve_wav_path(wav_path)
    hex_file = sample.get("hex_file", "?")
    dut._log.info(f"=== Audio phase: sample[{sample_idx}] {Path(wav_path).name} hex={hex_file} GT={gt_name} arith={arith_name} pytorch={pytorch_name} ===")
    if Path(wav_path) != resolved_wav_path:
        dut._log.info(f"  Resolved manifest WAV path to {resolved_wav_path}")

    pcm = _load_wav_pcm(resolved_wav_path)
    pdm_bits = _pcm_to_pdm(pcm)
    dut._log.info(f"  {len(pcm)} PCM samples -> {len(pdm_bits)} PDM bits")
    dut._log.info(f"  [initial state] {_probe_str(dut)}")

    handles = _resolve_probe_handles(dut)
    pulse_counts = {name: 0 for name in ['fft_sync', 'filterbank_done', 'log_done', 'mel_valid', 'spect_done', 'kws_start']}
    milestones = {name: False for name in ['cic_valid', 'fir_valid', 'fft_sync', 'fft_sync_aligned', 'fft_valid', 'power_valid', 'filterbank_done', 'log_done', 'mel_valid', 'spect_done', 'kws_start']}
    frontend_timeout_cycles = len(pdm_bits) + 200_000
    kws_timeout_cycles = len(pdm_bits) + 20_000_000
    log_every = int(os.getenv("KWS_LOG_EVERY", "500000"))
    t0_real = time.time()
    frontend_cyc = None
    kws_started_cyc = None
    rtl_class = None
    kws_done_seen = False

    for cyc in range(frontend_timeout_cycles):
        if cyc < len(pdm_bits):
            dut.input_PAD.value = 0b10 | (pdm_bits[cyc] & 1)
        else:
            dut.input_PAD.value = 0
        await RisingEdge(dut.clk_PAD)

        for name in pulse_counts:
            if _handle_is_one(handles[name]):
                pulse_counts[name] += 1
        for name in milestones:
            if not milestones[name] and _handle_is_one(handles[name]):
                milestones[name] = True
                dut._log.info(f"  [milestone] {name} observed at cyc={cyc + 1:,}")
                if name == 'kws_start':
                    kws_started_cyc = cyc + 1

        done, cls = _read_kws_pads(dut)
        if done:
            rtl_class = cls
            kws_done_seen = True
            frontend_cyc = cyc + 1
            break
        if kws_started_cyc is not None:
            frontend_cyc = cyc + 1
            break

        if cyc % log_every == log_every - 1:
            pdm_status = 'done' if cyc + 1 >= len(pdm_bits) else f'in flight ({cyc + 1}/{len(pdm_bits)})'
            dut._log.info(f"  [chip_top heartbeat] cyc={cyc + 1:,} sim={(cyc + 1) * CLK_PERIOD_NS / 1e6:.1f}ms real={time.time() - t0_real:.0f}s PDM={pdm_status} counts={pulse_counts} probes={_probe_str(dut)}")

    if frontend_cyc is None:
        missing = next((name for name, seen in milestones.items() if not seen), None)
        raise AssertionError(f"frontend never produced kws_start within {frontend_timeout_cycles:,} cycles; first missing milestone: {missing}; counts={pulse_counts}; probes={_probe_str(dut)}")

    if not kws_done_seen:
        remaining_pdm = pdm_bits[frontend_cyc:]
        if remaining_pdm:
            cocotb.start_soon(_drive_pdm_from_pads(dut, remaining_pdm))
            dut._log.info(f"  Continuing remaining PDM in background ({len(remaining_pdm):,} bits)")
        else:
            dut.input_PAD.value = 0

        remaining_timeout = max(1, kws_timeout_cycles - frontend_cyc)
        dut._log.info(f"  KWS started; waiting up to {remaining_timeout:,} more cycles for bidir_PAD[2]")
        done_cyc, cls = await _wait_for_kws_done_pad(dut, remaining_timeout)
        if done_cyc is None:
            raise AssertionError(f"KWS_DONE pad never asserted within {kws_timeout_cycles:,} total cycles; frontend_cyc={frontend_cyc:,}; probes={_probe_str(dut)}")
        rtl_class = cls
        kws_done_seen = True
        dut._log.info(f"  KWS_DONE pad observed after {done_cyc:,} post-start wait cycles")

    assert kws_done_seen, "KWS_DONE was not observed on bidir_PAD[2]"

    rtl_name = class_names[rtl_class] if rtl_class < len(class_names) else f"cls{rtl_class}"
    expected_class = arith_class if arith_class is not None else gt_class
    expected_name = arith_name if arith_class is not None else gt_name
    expected_source = "manifest arithmetic" if arith_class is not None else "dataset label"
    passed = rtl_class == expected_class
    scores, score_warning = _read_kws_scores(dut, class_names)

    dut._log.info(f"  Dataset label = {gt_name} ({gt_class})")
    dut._log.info(f"  Manifest arithmetic = {arith_name} ({arith_class})")
    dut._log.info(f"  Manifest PyTorch = {pytorch_name} ({pytorch_class})")
    dut._log.info(f"  RTL class from chip_top pads = {rtl_name} ({rtl_class})")
    dut._log.info(f"  Expected[{expected_source}] = {expected_name} ({expected_class})")
    if scores is not None:
        dut._log.info(f"  RTL GAP scores: {_format_kws_scores(scores)}")
        dut._log.info(f"  RTL GAP ranking: {_format_kws_score_ranking(scores)}")
    if score_warning:
        dut._log.warning(f"  RTL GAP score warning: {score_warning}")

    _append_sim_top_result(sample_idx=sample_idx, wav_path=resolved_wav_path, hex_file=hex_file, gt_name=gt_name, gt_class=gt_class, arith_name=arith_name, arith_class=arith_class, pytorch_name=pytorch_name, pytorch_class=pytorch_class, rtl_name=rtl_name, rtl_class=rtl_class, expected_source=expected_source, expected_name=expected_name, expected_class=expected_class, passed=passed, scores=scores, score_warning=score_warning, manifest_path=manifest_path)
    dut._log.info(f"  sim-top result appended to {RESULTS_TXT}")

    assert passed, f"chip_top RTL produced '{rtl_name}', expected '{expected_name}' from {expected_source}"
    dut._log.info("test_chip_top_e2e PASSED")


@cocotb.test()
async def test_chip_top_pad_smoke(dut):
    """Minimal RTL chip_top pad smoke."""

    logger = logging.getLogger("chip_top_tb")
    logger.info("Starting chip_top pad smoke")

    await start_up(dut)

    await ClockCycles(dut.clk_PAD, 10)
    assert _pad_bit(dut.bidir_PAD.value, 0) == "1", "UART RX pad should be held idle-high"
    assert _pad_bit(dut.bidir_PAD.value, 1) in {"0", "1"}, "UART TX pad should be known after reset"

    for pattern in (0x000, 0x001, 0x800, 0x5A5):
        dut.input_PAD.value = pattern
        await ClockCycles(dut.clk_PAD, 8)
        assert dut.input_PAD.value.to_unsigned() == pattern
        assert _pad_bit(dut.bidir_PAD.value, 1) in {"0", "1"}, "UART TX pad became X/Z"

    logger.info("chip_top pad smoke completed")


def link_readmemh_files(proj_path):
    rtl = (proj_path / "../src/rtl").resolve()
    links = [
        rtl / "STFFT/ZipCPU/hanning.hex",
        rtl / "Log-Mel/data/mel_coeffs_sparse.hex",
        rtl / "Log-Mel/data/mel_indices.hex",
        rtl / "Log-Mel/data/log2_lut.hex",
        rtl / "dscnn/bias_SRAM/bias.hex",
    ]
    # Icarus/vvp runs from sim_build under cocotb_tools.runner, while older
    # direct Makefile paths run from cocotb/. Put links in both places.
    link_dirs = [proj_path, proj_path / "sim_build"]
    for link_dir in link_dirs:
        link_dir.mkdir(exist_ok=True)
        for source in links:
            dest = link_dir / source.name
            try:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(source)
            except FileExistsError:
                pass


def rtl_sources(proj_path):
    src = (proj_path / "../src").resolve()
    rtl = src / "rtl"
    return [
        rtl / "flash/boot_pkg.sv",
        src / "chip_top.sv",
        src / "chip_core.sv",
        rtl / "top/full_pipeline_top.sv",
        rtl / "top/pipeline_top.sv",
        rtl / "dscnn/kws_top.sv",
        rtl / "dscnn/bias_SRAM/bias_SRAM.sv",
        rtl / "dscnn/feature_sram/feature_sram.sv",
        rtl / "dscnn/fsm/FSM.sv",
        rtl / "dscnn/mac_array/mac_array.sv",
        rtl / "dscnn/requant/requant.sv",
        rtl / "dscnn/spectrogram_sram/spectrogram_sram.sv",
        rtl / "dscnn/weight_sram/weight_sram.sv",
        rtl / "Log-Mel/rtl/log_top/logmel_top.sv",
        rtl / "Log-Mel/rtl/frame_control/frame_control.sv",
        rtl / "Log-Mel/rtl/log_lut/log_lut_sram.sv",
        rtl / "Log-Mel/rtl/log_lut/log_lut.sv",
        rtl / "Log-Mel/rtl/mac_unit/mac_unit.sv",
        rtl / "Log-Mel/rtl/mel_filterbank/mel_coeff_sram.sv",
        rtl / "Log-Mel/rtl/mel_filterbank/mel_filterbank.sv",
        rtl / "Log-Mel/rtl/output_buffer/output_buffer.sv",
        rtl / "Log-Mel/rtl/power_calc/power_calc.sv",
        rtl / "Log-Mel/rtl/spectral_vad/spectral_vad.sv",
        rtl / "Log-Mel/ip/Add.sv",
        rtl / "Log-Mel/ip/AddMopCsv.sv",
        rtl / "Log-Mel/ip/Cpr.sv",
        rtl / "Log-Mel/ip/Encode.sv",
        rtl / "Log-Mel/ip/FullAdder.sv",
        rtl / "Log-Mel/ip/LeadZeroDet.sv",
        rtl / "Log-Mel/ip/Log2.sv",
        rtl / "Log-Mel/ip/MulPPGenUns.sv",
        rtl / "Log-Mel/ip/MulUns.sv",
        rtl / "Log-Mel/ip/PrefixAnd.sv",
        rtl / "Log-Mel/ip/PrefixAndOr.sv",
        rtl / "Log-Mel/ip/SqrPPGenSgn.sv",
        rtl / "Log-Mel/ip/SqrSgn.sv",
        rtl / "STFFT/rtl/stfft.sv",
        rtl / "STFFT/rtl/fft_twiddle_rom.sv",
        rtl / "STFFT/rtl/fft_data_ram.sv",
        rtl / "STFFT/R2FFT/hdl/R2FFT.sv",
        rtl / "STFFT/R2FFT/hdl/fft.sv",
        rtl / "STFFT/R2FFT/hdl/bfp_bitWidthAcc.sv",
        rtl / "STFFT/R2FFT/hdl/bfp_bitWidthDetector.sv",
        rtl / "STFFT/R2FFT/hdl/bfp_maxBitWidth.sv",
        rtl / "STFFT/R2FFT/hdl/bfp_Shifter.sv",
        rtl / "STFFT/R2FFT/hdl/bitReverseCounter.sv",
        rtl / "STFFT/R2FFT/hdl/butterflyCore.sv",
        rtl / "STFFT/R2FFT/hdl/butterflyUnit.sv",
        rtl / "STFFT/R2FFT/hdl/fftAddressGenerator.sv",
        rtl / "STFFT/R2FFT/hdl/radix2Butterfly.sv",
        rtl / "STFFT/R2FFT/hdl/ramPipelineBridge.sv",
        rtl / "STFFT/R2FFT/hdl/rwBusMux.sv",
        rtl / "STFFT/R2FFT/hdl/twiddleFactorRomBridge.sv",
        rtl / "flash/boot_controller.sv",
        rtl / "flash/features_boot_router.sv",
        rtl / "flash/dscnn_boot_router.sv",
        rtl / "flash/uart_rx.v",
        rtl / "flash/uart_tx.v",
        rtl / "flash/uart.v",
        rtl / "CIC/cic.sv",
        rtl / "SPECT_BUFFER/spect_buffer_ctrl.sv",
        rtl / "FIR/rtl/compFIR.sv",
    ]


def chip_top_runner():

    proj_path = Path(__file__).resolve().parent

    sources = []
    defines = {f"SLOT_{slot.upper()}": True, "SIM": True}
    includes = [proj_path / "../src/"]

    if gl:
        # SCL models
        sources.append(Path(pdk_root) / pdk / "libs.ref" / scl / "verilog" / f"{scl}.v")
        sources.append(Path(pdk_root) / pdk / "libs.ref" / scl / "verilog" / "primitives.v")

        # We use the powered netlist
        sources.append(proj_path / f"../final/pnl/{hdl_toplevel}.pnl.v")

        defines = {"FUNCTIONAL": True, "USE_POWER_PINS": True}
    else:
        # boot_pkg.sv must be first so Icarus can resolve boot_bus_t before chip_core.sv.
        sources.extend(rtl_sources(proj_path))

    sources += [
        # IO pad models
        Path(pdk_root) / pdk / "libs.ref/gf180mcu_fd_io/verilog/gf180mcu_fd_io.v",
        Path(pdk_root) / pdk / "libs.ref/gf180mcu_fd_io/verilog/gf180mcu_ws_io.v",

        # Custom IP
        proj_path / "../ip/gf180mcu_ws_ip__id/vh/gf180mcu_ws_ip__id.v",
        proj_path / "../ip/gf180mcu_ws_ip__logo/vh/gf180mcu_ws_ip__logo.v",
    ]

    build_args = []

    if sim == "icarus":
        # For debugging
        # build_args = ["-Winfloop", "-pfileline=1"]
        pass

    if sim == "verilator":
        build_args = ["--timing"]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel=hdl_toplevel,
        defines=defines,
        always=True,
        includes=includes,
        build_args=build_args,
        waves=False,
    )
    link_readmemh_files(proj_path)

    plusargs = []

    runner.test(
        hdl_toplevel=hdl_toplevel,
        test_module="chip_top_tb,",
        plusargs=plusargs,
        waves=False,
    )


if __name__ == "__main__":
    chip_top_runner()
