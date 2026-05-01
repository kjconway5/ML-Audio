# SPDX-FileCopyrightText: © 2025 XXX Authors
# SPDX-License-Identifier: Apache-2.0

"""
chip_core_tb.py
End-to-end cocotb testbench for chip_core.sv

DUT: chip_core (1×1 slot — NUM_INPUT_PADS=12, NUM_BIDIR_PADS=40, NUM_ANALOG_PADS=2)

Pad / signal mapping
--------------------
  bidir_in[0]  → UART_RX   : test bit-bangs UART packets to boot the chip
  bidir_out[1] → UART_TX   : chip sends ACK/NACK bytes back to the test
  bidir_out[2] → KWS_DONE  : pulses high for one cycle when inference completes
  bidir_out[5:3] → KWS_CLASS[2:0] : winning class index (valid while DONE is high)

  input_in[0]  → PDM_DATA  : 1-bit PDM stream (1 = positive, 0 = negative)
  input_in[1]  → PDM_VALID : strobe — one pulse per PDM bit (same convention as
                              test_full_pipeline_top.py's valid_i)

Tests
-----
  1. test_chip_core_boot   — mini UART boot with synthetic data; verifies ACK
                             responses, boot_done release, and pad OE directions
  2. test_chip_core_e2e    — full boot with real model weights + mel coefficients,
                             drives PDM audio derived from a speech-commands WAV,
                             and verifies the KWS class output on the bidir pads
"""

import json, os, sys, time
import cocotb
from cocotb.clock    import Clock
from cocotb.triggers import RisingEdge, ClockCycles, First, Timer
from pathlib         import Path

# Slot parameters (1×1)
NUM_INPUT_PADS = 12
NUM_BIDIR_PADS = 40

# Pad indices — must match chip_core.sv localparam
UART_RX_PAD    = 0
UART_TX_PAD    = 1
KWS_DONE_PAD   = 2
KWS_CLASS_BASE = 3   # class[0]=pad3, class[1]=pad4, class[2]=pad5

PDM_DATA_PAD   = 0   # input_in[0]
PDM_VALID_PAD  = 1   # input_in[1]

# Clock / UART timing
#   Chip clock  = 25 MHz  → CLK_PERIOD_NS = 40 ns
#   RTL uses prescale=1 under `ifdef SIM → 8 cycles/bit (27× faster than real)
#   Real UART prescale=27 → 216 cycles/bit at 115200 baud
CLK_PERIOD_NS = 40
UART_PRESCALE = 1    # matches `ifdef SIM in chip_core.sv
BIT_CYCLES    = UART_PRESCALE * 8   # 8 cycles/bit in sim

# Boot protocol constants — must match boot_pkg.sv
SYNC_0    = 0xAA
SYNC_1    = 0x55
ACK_BYTE  = 0x06
NACK_BYTE = 0xEE

MOD_FEATURES = 0x0
MOD_DSCNN    = 0x1
MOD_CONTROL  = 0xF

FEAT_LOG_LUT   = 0x0   # 16-bit words
FEAT_MEL_COEFF = 0x1   # 16-bit words
FEAT_MEL_META  = 0x2   # 8-bit bytes

DSCNN_WEIGHTS  = 0x0   # 8-bit bytes
DSCNN_CFG      = 0x1   # 8-bit bytes

CTRL_BOOT_DONE = 0x0

def _make_target(mod, sub):
    return ((mod & 0xF) << 4) | (sub & 0xF)


_SRC       = Path(__file__).resolve().parent.parent / "src"
_ML        = _SRC / "ml"
_RTL       = _SRC / "rtl"
_KWS_DIR   = _RTL / "dscnn/kws_top"
_LOGMEL    = _RTL / "Log-Mel/data"
_MODEL_DIR = _ML / "models/dscnn-32full-v1"

LOG_LUT_HEX   = _LOGMEL    / "log2_lut.hex"
MEL_COEFF_HEX = _LOGMEL    / "mel_coeffs_sparse.hex"
MEL_INDEX_HEX = _LOGMEL    / "mel_indices.hex"
WEIGHTS_HEX   = _MODEL_DIR / "weights.hex"
SCALES_TXT    = _MODEL_DIR / "scales.txt"
MANIFEST_JSON = _KWS_DIR   / "test_vectors.json"
SPECT_DIR     = _KWS_DIR   / "spectrograms"
MODEL_FILTERS = 32


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

# Hex file loader

def _load_hex(path, signed=False, width=8):
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            v = int(s, 16)
            if signed and v >= (1 << (width - 1)):
                v -= (1 << width)
            vals.append(v)
    return vals

# Layer-config packing
#   Replicates program_layers() from test_kws_top.py but produces flat byte
#   arrays suitable for UART packet payloads.

sys.path.insert(0, str(_KWS_DIR))
from rtl_golden import load_layer_cfgs   # noqa: E402  (import after path insert)

def _pack_layer_cfgs(layer_cfgs):
    """Return (field_bytes[0x00..0x9F], mult_bytes[0xA0..0xC7])."""
    buf = [0] * 0xC8

    for (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
         dw, w_off, mult, shift, relu, ofmap_h, ofmap_w, bias_off) in layer_cfgs:

        b = layer << 4
        buf[b | 0x0] = in_ch   & 0xFF
        buf[b | 0x1] = out_ch  & 0xFF
        buf[b | 0x2] = kH      & 0xFF
        buf[b | 0x3] = kW      & 0xFF
        buf[b | 0x4] = sh      & 0xFF
        buf[b | 0x5] = sw      & 0xFF
        buf[b | 0x6] = ph      & 0xFF
        buf[b | 0x7] = pw      & 0xFF
        buf[b | 0x8] = dw      & 0xFF
        buf[b | 0x9] = w_off         & 0xFF
        buf[b | 0xA] = (w_off >> 8)  & 0x1F
        buf[b | 0xB] = shift         & 0xFF
        buf[b | 0xC] = (relu & 0x1) | ((bias_off >> 7) & 0x2)
        buf[b | 0xD] = ofmap_h & 0xFF
        buf[b | 0xE] = ofmap_w & 0xFF
        buf[b | 0xF] = bias_off & 0xFF

        mb = 0xA0 + layer * 4
        buf[mb + 0] = (mult >>  0) & 0xFF
        buf[mb + 1] = (mult >>  8) & 0xFF
        buf[mb + 2] = (mult >> 16) & 0xFF
        buf[mb + 3] = (mult >> 24) & 0xFF

    return buf[0x00:0xA0], buf[0xA0:0xC8]

# UART helpers

async def _uart_tx_byte(dut, byte_val, log=None, label=""):
    """Drive one UART byte on bidir_in[UART_RX_PAD] (start + 8 data + stop)."""
    if log:
        log.debug(f"  TX byte {label}0x{byte_val:02X}")
    # Start bit (low)
    dut.bidir_in.value = 0
    await ClockCycles(dut.clk, BIT_CYCLES)
    # 8 data bits, LSB first
    for i in range(8):
        dut.bidir_in.value = (byte_val >> i) & 1
        await ClockCycles(dut.clk, BIT_CYCLES)
    # Stop bit (high = idle)
    dut.bidir_in.value = 1
    await ClockCycles(dut.clk, BIT_CYCLES)


def _bidir_bit(dut, idx):
    """
    Read a single bit of bidir_out by position without converting the whole
    40-bit word to int (which fails if any OTHER bit is X, e.g. kws_done
    while kws_top is in inference_reset).  Returns the integer bit value,
    or the supplied default if that specific bit is X.
    """
    binstr = dut.bidir_out.value.binstr   # MSB-first string, length=NUM_BIDIR_PADS
    ch = binstr[-(idx + 1)]               # index from the right
    if ch == '1':
        return 1
    if ch == '0':
        return 0
    return None   # X or Z


def _tx_bit(dut):
    """Read bidir_out[UART_TX_PAD]; return 1 for X/Z (UART idles high)."""
    b = _bidir_bit(dut, UART_TX_PAD)
    return b if b is not None else 1


async def _uart_rx_byte(dut, timeout_cycles=800_000):
    """
    Receive one UART byte from bidir_out[UART_TX_PAD].
    Polls for the falling start-bit edge then samples each data bit at mid-point.
    Reads bit UART_TX_PAD individually (not int() of the whole bus) so that X
    on other bits (e.g. kws_done during boot) does not block reception.
    """
    log = dut._log
    log.info("  [UART RX] waiting for start bit on bidir_out[1] (chip TX)...")
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if _tx_bit(dut) == 0:
            log.info(f"  [UART RX] start bit detected at poll cycle {cyc}")
            break
        if cyc % 50000 == 49999:
            log.info(f"  [UART RX] still idle at poll cycle {cyc}, TX bit={_tx_bit(dut)}")
    else:
        raise AssertionError("UART RX: timeout waiting for chip TX start bit")

    # Skip to middle of first data bit (1.5 × BIT_CYCLES from start-bit leading edge)
    await ClockCycles(dut.clk, BIT_CYCLES + BIT_CYCLES // 2)

    byte_val = 0
    for i in range(8):
        byte_val |= (_tx_bit(dut) << i)
        if i < 7:
            await ClockCycles(dut.clk, BIT_CYCLES)

    await ClockCycles(dut.clk, BIT_CYCLES)   # consume stop bit
    log.info(f"  [UART RX] received byte 0x{byte_val:02X}")
    return byte_val


def _build_packet(target, addr, payload):
    """Return flat byte list: [AA, 55, target, addr_hi, addr_lo, len_hi, len_lo, ...payload..., cksum]."""
    payload = list(payload)
    n       = len(payload)
    body    = [target, (addr >> 8) & 0xFF, addr & 0xFF,
               (n >> 8) & 0xFF, n & 0xFF] + payload
    cksum   = 0
    for b in body:
        cksum ^= b
    return [SYNC_0, SYNC_1] + body + [cksum & 0xFF]


async def _send_packet(dut, target, addr, payload):
    """Frame, send, and wait for ACK on a single boot packet."""
    log = dut._log
    pkt = _build_packet(target, addr, payload)
    log.info(f"  [PKT] target=0x{target:02X} addr=0x{addr:04X} payload_len={len(payload)} total_bytes={len(pkt)}")
    for idx, b in enumerate(pkt):
        if idx < 7 or idx == len(pkt) - 1:  # header + checksum only, skip bulk payload
            log.info(f"    TX[{idx}] = 0x{b:02X}")
        elif idx == 7:
            log.info(f"    TX[7..{len(pkt)-2}] = payload ({len(payload)} bytes) ...")
        await _uart_tx_byte(dut, b)
    log.info(f"  [PKT] all bytes sent, awaiting ACK ...")
    resp = await _uart_rx_byte(dut)
    assert resp == ACK_BYTE, \
        f"Expected ACK 0x{ACK_BYTE:02X} for target=0x{target:02X} addr=0x{addr:04X}, " \
        f"got 0x{resp:02X}"
    log.info(f"  [PKT] ACK received OK")

# Boot sequence

async def _boot_chip(dut, mini=False):
    """
    Load all memories via UART packets and assert boot_done.

    mini=True: use tiny synthetic data for a fast protocol smoke-test.
    mini=False: load real model weights, mel coefficients, and layer configs.
    """
    if mini:
        lut_words    = [(i * 0x0111) & 0xFFFF for i in range(64)]
        mel_words    = [(i * 0x0303) & 0xFFFF for i in range(10)]
        meta_bytes   = [(i * 3)      & 0xFF   for i in range(10)]
        weight_bytes = [(i * 7)      & 0xFF   for i in range(20)]
        cfg_fields   = [0] * 160
        cfg_mults    = [0] * 40
    else:
        lut_words    = _load_hex(LOG_LUT_HEX,   signed=False, width=16)
        mel_words    = _load_hex(MEL_COEFF_HEX,  signed=False, width=16)
        meta_bytes   = _load_hex(MEL_INDEX_HEX,  signed=False, width=8)
        weight_bytes = _load_hex(WEIGHTS_HEX,    signed=True,  width=8)
        layer_cfgs   = load_layer_cfgs(SCALES_TXT, n_filters=32)
        cfg_fields, cfg_mults = _pack_layer_cfgs(layer_cfgs)

    # Log LUT — 16-bit words packed lo/hi
    lut_payload = []
    for w in lut_words:
        lut_payload += [w & 0xFF, (w >> 8) & 0xFF]
    await _send_packet(dut, _make_target(MOD_FEATURES, FEAT_LOG_LUT), 0, lut_payload)
    dut._log.info(f"  Log LUT loaded ({len(lut_words)} words)")

    # Mel coefficients — 16-bit words packed lo/hi
    mel_payload = []
    for w in mel_words:
        mel_payload += [w & 0xFF, (w >> 8) & 0xFF]
    await _send_packet(dut, _make_target(MOD_FEATURES, FEAT_MEL_COEFF), 0, mel_payload)
    dut._log.info(f"  Mel coeffs loaded ({len(mel_words)} words)")

    # Mel meta / indices — 8-bit bytes
    await _send_packet(dut, _make_target(MOD_FEATURES, FEAT_MEL_META), 0, meta_bytes)
    dut._log.info(f"  Mel indices loaded ({len(meta_bytes)} bytes)")

    # DS-CNN weights — 8-bit bytes
    w_payload = [b & 0xFF for b in weight_bytes]
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_WEIGHTS), 0, w_payload)
    dut._log.info(f"  Weights loaded ({len(weight_bytes)} bytes)")

    # DS-CNN layer config: field registers (addresses 0x00 – 0x9F)
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_CFG), 0x00, cfg_fields)
    # DS-CNN layer config: multiply-shift registers (addresses 0xA0 – 0xC7)
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_CFG), 0xA0, cfg_mults)
    # Set cfg_load_done (address 0xFF, value 0x01)
    await _send_packet(dut, _make_target(MOD_DSCNN, DSCNN_CFG), 0xFF, [0x01])
    dut._log.info("  Layer configs loaded, cfg_load_done set")

    # Release inference reset
    await _send_packet(dut, _make_target(MOD_CONTROL, CTRL_BOOT_DONE), 0, [])
    dut._log.info("  boot_done asserted — pipeline + KWS out of reset")

    await ClockCycles(dut.clk, 10)

# Reset

async def _do_reset(dut):
    dut.rst_n.value    = 0
    dut.bidir_in.value = 1   # UART RX idle (high)
    dut.input_in.value = 0
    await ClockCycles(dut.clk, 20)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)

# PDM audio helpers  (mirrors pcm_to_pdm / drive_pdm in test_full_pipeline_top.py)

def _pcm_to_pdm(pcm, decim=63):
    """Software sigma-delta modulator: PCM samples → PDM bitstream."""
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
    return pdm


async def _drive_pdm(dut, pdm_bits):
    """
    Drive PDM bits through input_in[1:0].
      bit 0 = PDM_DATA  (the PDM bit value)
      bit 1 = PDM_VALID (strobe — always 1 while driving audio)
    One PDM bit per chip-clock cycle, identical to test_full_pipeline_top.py.
    """
    for b in pdm_bits:
        dut.input_in.value = 0b10 | (b & 1)   # valid=1, data=b
        await RisingEdge(dut.clk)
    dut.input_in.value = 0   # deassert valid


def _load_wav_pcm(wav_path, target_samples=7_520):
    """Load a WAV file and return int32 PCM trimmed / zero-padded to target_samples."""
    import numpy as np
    try:
        from scipy.io import wavfile
        rate, data = wavfile.read(str(wav_path))
    except Exception:
        import soundfile as sf
        data, rate = sf.read(str(wav_path))
        data = (data * 32767).astype("int16")

    data = np.asarray(data)
    if data.ndim == 2:
        data = data[:, 0]
    if data.dtype == "int16":
        pcm = data.astype("int32")
    elif data.dtype == "float32" or data.dtype == "float64":
        pcm = (data * 32767).astype("int32")
    else:
        pcm = data.astype("int32")

    # Resample if needed
    if rate != 16_000:
        from scipy.signal import resample_poly
        from math import gcd
        g   = gcd(16_000, int(rate))
        pcm = resample_poly(pcm, 16_000 // g, rate // g).astype("int32")

    # Trim / pad
    if len(pcm) >= target_samples:
        return pcm[:target_samples]
    import numpy as np
    return np.concatenate([pcm, np.zeros(target_samples - len(pcm), dtype="int32")])


def _select_manifest_sample(samples):
    """
    Pick the WAV sample for the full-chip test.

    Environment controls:
      KWS_SAMPLE_INDEX=<n>     Select manifest sample index n.
      KWS_SAMPLE_MATCH=<text>  Select first sample whose wav path or GT name contains text.
      KWS_KEYWORD=<label>      Restrict selection to a ground-truth label.

    If neither is set, preserves the historical behavior: samples[0].
    """
    if not samples:
        raise AssertionError("test_vectors.json contains no samples")

    sample_keyword = os.getenv("KWS_KEYWORD")
    sample_index = os.getenv("KWS_SAMPLE_INDEX")
    sample_match = os.getenv("KWS_SAMPLE_MATCH")

    filtered = list(enumerate(samples))
    if sample_keyword:
        keyword = sample_keyword.lower()
        filtered = [
            (idx, sample)
            for idx, sample in filtered
            if str(sample.get("ground_truth_name", "")).lower() == keyword
        ]
        if not filtered:
            available = sorted({str(s.get("ground_truth_name", "?")) for s in samples})
            raise AssertionError(
                f"KWS_KEYWORD={sample_keyword!r} did not match this manifest. "
                f"Available ground_truth_name values: {available}. "
                f"Regenerate test vectors with generate_spect_full.py --keyword {sample_keyword} "
                f"or point KWS_MANIFEST_JSON at the right test_vectors.json."
            )

    if sample_index is not None:
        try:
            idx = int(sample_index, 0)
        except ValueError as exc:
            raise AssertionError(f"KWS_SAMPLE_INDEX must be an integer, got {sample_index!r}") from exc
        if idx < 0 or idx >= len(samples):
            raise AssertionError(
                f"KWS_SAMPLE_INDEX={idx} is out of range; manifest has "
                f"indices 0..{len(samples)-1}"
            )
        if sample_keyword and (idx, samples[idx]) not in filtered:
            raise AssertionError(
                f"KWS_SAMPLE_INDEX={idx} exists, but its ground_truth_name is "
                f"{samples[idx].get('ground_truth_name')!r}, not KWS_KEYWORD={sample_keyword!r}"
            )
        return idx, samples[idx]

    if sample_match:
        needle = sample_match.lower()
        for idx, sample in filtered:
            wav = str(sample.get("wav", "")).lower()
            gt = str(sample.get("ground_truth_name", "")).lower()
            if needle in wav or needle in gt:
                return idx, sample
        raise AssertionError(
            f"KWS_SAMPLE_MATCH={sample_match!r} did not match any wav path or "
            f"ground_truth_name in the selected manifest samples"
        )

    return filtered[0]

# KWS output reader

def _read_kws_pads(dut):
    def _safe(idx):
        b = _bidir_bit(dut, idx)
        return b if b is not None else 0   # X → 0 (not done / class 0)
    kws_done  = _safe(KWS_DONE_PAD)
    kws_class = (_safe(KWS_CLASS_BASE+2) << 2) | (_safe(KWS_CLASS_BASE+1) << 1) | _safe(KWS_CLASS_BASE)
    return kws_done, kws_class

# Internal-signal probe for debugging

def _probe(dut):
    """
    Sample key chip_core internal signals.
    Returns a dict; value is int/str, '?' if signal doesn't exist in this build.
    """
    def _get(path):
        try:
            obj = dut
            for part in path.split('.'):
                obj = getattr(obj, part)
            v = obj.value
            if hasattr(v, 'is_resolvable'):
                return int(v) if v.is_resolvable else 'X'
            return int(v)
        except Exception:
            return '?'

    return {
        'boot_done':        _get('u_boot_ctrl.boot_done_o'),
        'inference_reset':  _get('inference_reset'),
        'spect_done':       _get('spect_done'),
        'kws_start':        _get('kws_start'),
        'kws_done':         _get('kws_done'),
        'pdm_valid':        _get('pdm_valid'),
        # KWS FSM state/progress (kws_top instantiates FSM as inst_ctrl)
        'fsm_state':        _get('kws_inst.inst_ctrl.state'),
        'kws_layer':        _get('kws_inst.inst_ctrl.layer'),
        'kws_oh':           _get('kws_inst.inst_ctrl.oh'),
        'kws_ow':           _get('kws_inst.inst_ctrl.ow'),
        'kws_oc':           _get('kws_inst.inst_ctrl.oc'),
        'kws_ic':           _get('kws_inst.inst_ctrl.ic'),
        'kws_kh':           _get('kws_inst.inst_ctrl.kh'),
        'kws_kw':           _get('kws_inst.inst_ctrl.kw'),
        'cfg_load_done':    _get('kws_inst.inst_ctrl.cfg_load_done'),
        'spect_ready':      _get('kws_inst.inst_ctrl.spect_ready'),
        'bias_addr':        _get('kws_inst.bias_addr'),
        'w_raddr':          _get('kws_inst.w_raddr'),
        'mac_en':           _get('kws_inst.mac_en'),
        'mac_clear':        _get('kws_inst.mac_clear'),
        'mac_acc':          _get('kws_inst.mac_acc'),
        'rq_out':           _get('kws_inst.rq_out'),
        # spect_buffer write-select (tells us which spectrogram bank is active)
        'spect_write_sel':  _get('spect_write_sel'),
        # Pipeline progress probes. These identify where the front-end stalls
        # before KWS can ever see a complete spectrogram.
        'cic_valid':        _get('pipeline_inst.cic_valid'),
        'fir_valid':        _get('pipeline_inst.fir_valid_o'),
        'fft_sync':         _get('pipeline_inst.u_stfft.o_fft_sync'),
        'fft_sync_aligned': _get('pipeline_inst.fft_sync_aligned'),
        'fft_valid':        _get('pipeline_inst.fft_valid'),
        'power_valid':      _get('pipeline_inst.u_logmel.power_valid'),
        'filterbank_done':  _get('pipeline_inst.u_logmel.filterbank_done'),
        'log_done':         _get('pipeline_inst.u_logmel.log_done'),
        'mel_valid':        _get('pipeline_inst.mel_valid'),
        'mel_data':         _get('pipeline_inst.mel_compensated'),
        'spect_wr_addr':    _get('pipeline_inst.u_spect_buf.wr_addr'),
    }


def _probe_str(dut):
    p = _probe(dut)
    return (
        f"boot_done={p['boot_done']}  inf_rst={p['inference_reset']}  "
        f"pdm_valid={p['pdm_valid']}  spect_done={p['spect_done']}  "
        f"spect_sel={p['spect_write_sel']}  kws_start={p['kws_start']}  "
        f"kws_done={p['kws_done']}  fsm={p['fsm_state']}  "
        f"cic={p['cic_valid']} fir={p['fir_valid']} "
        f"fft_sync={p['fft_sync']}/{p['fft_sync_aligned']} fft_valid={p['fft_valid']} "
        f"pwr={p['power_valid']} fb_done={p['filterbank_done']} "
        f"log_done={p['log_done']} mel={p['mel_valid']} wr={p['spect_wr_addr']}"
    )


def _kws_probe_str(dut):
    p = _probe(dut)
    return (
        f"fsm={p['fsm_state']} layer={p['kws_layer']} "
        f"oh={p['kws_oh']} ow={p['kws_ow']} oc={p['kws_oc']} "
        f"ic={p['kws_ic']} kh={p['kws_kh']} kw={p['kws_kw']} "
        f"cfg_done={p['cfg_load_done']} spect_ready={p['spect_ready']} "
        f"bias_addr={p['bias_addr']} waddr={p['w_raddr']} "
        f"mac_en={p['mac_en']} mac_clear={p['mac_clear']} "
        f"mac_acc={p['mac_acc']} rq_out={p['rq_out']}"
    )


def _resolve_probe_handles(dut):
    paths = {
        'cic_valid':        'pipeline_inst.cic_valid',
        'fir_valid':        'pipeline_inst.fir_valid_o',
        'fft_sync':         'pipeline_inst.u_stfft.o_fft_sync',
        'fft_sync_aligned': 'pipeline_inst.fft_sync_aligned',
        'fft_valid':        'pipeline_inst.fft_valid',
        'power_valid':      'pipeline_inst.u_logmel.power_valid',
        'filterbank_done':  'pipeline_inst.u_logmel.filterbank_done',
        'log_done':         'pipeline_inst.u_logmel.log_done',
        'mel_valid':        'pipeline_inst.mel_valid',
        'spect_done':       'spect_done',
        'kws_start':        'kws_start',
        'kws_done':         'kws_done',
        'mel_data':         'pipeline_inst.mel_compensated',
        'spect_wr_addr':    'pipeline_inst.u_spect_buf.wr_addr',
    }
    handles = {}
    for name, path in paths.items():
        try:
            obj = dut
            for part in path.split('.'):
                obj = getattr(obj, part)
            handles[name] = obj
        except Exception:
            handles[name] = None
    return handles


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
    if handle is None:
        return False
    try:
        value = handle.value
        if hasattr(value, 'is_resolvable') and not value.is_resolvable:
            return False
        return int(value) == 1
    except Exception:
        return False


def _handle_bits(handle, default='?'):
    if handle is None:
        return default
    try:
        return str(handle.value)
    except Exception:
        return default


def _handle_has_x(handle):
    bits = _handle_bits(handle, default='')
    return 'x' in bits.lower() or 'z' in bits.lower()


def _missing_milestone(milestones):
    for name, seen in milestones.items():
        if not seen:
            return name
    return None


async def _monitor_kws_progress(dut, start_cyc, done_handle, log_every_cycles):
    elapsed = 0
    while True:
        await ClockCycles(dut.clk, log_every_cycles)
        elapsed += log_every_cycles
        dut._log.info(
            f"  [KWS progress] since_start={elapsed:,} "
            f"approx_cyc={start_cyc + elapsed:,} {_kws_probe_str(dut)}"
        )
        if _handle_is_one(done_handle):
            return


def _resolve_kws_x_handles(dut):
    paths = {
        'state':          'kws_inst.inst_ctrl.state',
        'layer':          'kws_inst.inst_ctrl.layer',
        'oh':             'kws_inst.inst_ctrl.oh',
        'ow':             'kws_inst.inst_ctrl.ow',
        'oc':             'kws_inst.inst_ctrl.oc',
        'ic':             'kws_inst.inst_ctrl.ic',
        'kh':             'kws_inst.inst_ctrl.kh',
        'kw':             'kws_inst.inst_ctrl.kw',
        'buf_sel':        'kws_inst.inst_ctrl.buf_sel',
        'spect_read_sel': 'kws_inst.inst_ctrl.spect_read_sel',
        'comb_in_bounds': 'kws_inst.inst_ctrl.comb_in_bounds',
        'comb_sp_raddr':  'kws_inst.inst_ctrl.comb_sp_raddr',
        'comb_feat_addr': 'kws_inst.inst_ctrl.comb_feat_addr',
        'comb_w_addr':    'kws_inst.inst_ctrl.comb_w_addr',
        'sp_raddr':       'kws_inst.ss_a_raddr',
        'sp_a_rdata':     'kws_inst.ss_a_rdata',
        'sp_b_rdata':     'kws_inst.ss_b_rdata',
        'fs_a_we':        'kws_inst.fs_a_we',
        'fs_b_we':        'kws_inst.fs_b_we',
        'fs_a_waddr':     'kws_inst.fs_a_waddr',
        'fs_b_waddr':     'kws_inst.fs_b_waddr',
        'fs_a_wdata':     'kws_inst.fs_a_wdata',
        'fs_b_wdata':     'kws_inst.fs_b_wdata',
        'fs_a_raddr':     'kws_inst.fs_a_raddr',
        'fs_b_raddr':     'kws_inst.fs_b_raddr',
        'fs_a_rdata':     'kws_inst.fs_a_rdata',
        'fs_b_rdata':     'kws_inst.fs_b_rdata',
        'w_raddr':        'kws_inst.w_raddr',
        'w_rdata':        'kws_inst.w_rdata',
        'bias_addr':      'kws_inst.bias_addr',
        'bias_data':      'kws_inst.bias_data',
        'mac_en':         'kws_inst.mac_en',
        'mac_clear':      'kws_inst.mac_clear',
        'mac_ifmap':      'kws_inst.mac_ifmap',
        'mac_weight':     'kws_inst.mac_weight',
        'mac_bias':       'kws_inst.mac_bias',
        'mac_acc':        'kws_inst.mac_acc',
        'rq_out':         'kws_inst.rq_out',
    }
    handles = {}
    for name, path in paths.items():
        try:
            obj = dut
            for part in path.split('.'):
                obj = getattr(obj, part)
            handles[name] = obj
        except Exception:
            handles[name] = None
    return handles


def _kws_x_context(h):
    return (
        f"state={_handle_int(h['state'])} layer={_handle_int(h['layer'])} "
        f"oh={_handle_int(h['oh'])} ow={_handle_int(h['ow'])} "
        f"oc={_handle_int(h['oc'])} ic={_handle_int(h['ic'])} "
        f"kh={_handle_int(h['kh'])} kw={_handle_int(h['kw'])} "
        f"buf_sel={_handle_int(h['buf_sel'])} spect_read_sel={_handle_int(h['spect_read_sel'])} "
        f"in_bounds={_handle_int(h['comb_in_bounds'])} "
        f"sp_addr={_handle_int(h['comb_sp_raddr'])}/{_handle_int(h['sp_raddr'])} "
        f"feat_addr={_handle_int(h['comb_feat_addr'])} "
        f"waddr={_handle_int(h['comb_w_addr'])}/{_handle_int(h['w_raddr'])} "
        f"bias_addr={_handle_int(h['bias_addr'])}"
    )


def _kws_x_values(h):
    return (
        f"sp_a={_handle_bits(h['sp_a_rdata'])} sp_b={_handle_bits(h['sp_b_rdata'])} "
        f"fs_a_raddr={_handle_int(h['fs_a_raddr'])} fs_a={_handle_bits(h['fs_a_rdata'])} "
        f"fs_b_raddr={_handle_int(h['fs_b_raddr'])} fs_b={_handle_bits(h['fs_b_rdata'])} "
        f"w={_handle_bits(h['w_rdata'])} bias={_handle_bits(h['bias_data'])} "
        f"mac_ifmap={_handle_bits(h['mac_ifmap'])} "
        f"mac_weight={_handle_bits(h['mac_weight'])} mac_bias={_handle_bits(h['mac_bias'])} "
        f"mac_acc={_handle_bits(h['mac_acc'])} rq_out={_handle_bits(h['rq_out'])}"
    )


async def _monitor_kws_x_sources(dut, done_handle, log_limit=24, fail_fast=False):
    h = _resolve_kws_x_handles(dut)
    log_count = 0
    suppressed = False

    def _report(tag, cyc, names):
        nonlocal log_count, suppressed
        details = ", ".join(f"{name}={_handle_bits(h[name])}" for name in names)
        if log_count < log_limit:
            dut._log.warning(
                f"  [KWS X SOURCE] {tag} cyc_since_monitor={cyc:,} "
                f"{details}; {_kws_x_context(h)}; {_kws_x_values(h)}"
            )
            log_count += 1
        elif not suppressed:
            dut._log.warning(
                f"  [KWS X SOURCE] suppressing further X-source logs after "
                f"{log_limit} events; latest context: {_kws_x_context(h)}"
            )
            suppressed = True
        if fail_fast:
            raise AssertionError(
                f"KWS X detected at {tag}: {details}; {_kws_x_context(h)}"
            )

    cyc = 0
    while True:
        await RisingEdge(dut.clk)
        cyc += 1

        if _handle_is_one(h['mac_en']):
            mac_inputs = ['mac_ifmap', 'mac_weight', 'mac_bias']
            bad = [name for name in mac_inputs if _handle_has_x(h[name])]
            if bad:
                _report("MAC input", cyc, bad)

            source_names = ['w_rdata']
            if _handle_int(h['layer'], default=0) == 0:
                if _handle_int(h['spect_read_sel'], default=0) == 0:
                    source_names.append('sp_a_rdata')
                else:
                    source_names.append('sp_b_rdata')
            else:
                if _handle_int(h['buf_sel'], default=0) == 0:
                    source_names.append('fs_a_rdata')
                else:
                    source_names.append('fs_b_rdata')
            bad = [name for name in source_names if _handle_has_x(h[name])]
            if bad:
                _report("MAC source read", cyc, bad)

        if _handle_int(h['state']) == 2 or _handle_is_one(h['mac_clear']):
            bad = [name for name in ['bias_data', 'mac_bias'] if _handle_has_x(h[name])]
            if bad:
                _report("bias load", cyc, bad)

        if _handle_int(h['state']) == 6 or _handle_is_one(h['fs_a_we']) or _handle_is_one(h['fs_b_we']):
            writeback = ['rq_out']
            if _handle_is_one(h['fs_a_we']):
                writeback.append('fs_a_wdata')
            if _handle_is_one(h['fs_b_we']):
                writeback.append('fs_b_wdata')
            bad = [name for name in writeback if _handle_has_x(h[name])]
            if bad:
                _report("feature writeback", cyc, bad)

        if _handle_is_one(done_handle):
            return



# Test 2 — full end-to-end KWS

@cocotb.test()
async def test_chip_core_e2e(dut):
    """
    Full end-to-end verification:
      1. Load all real weights, mel coefficients, and layer configs via UART.
      2. Drive PDM audio (from a speech-commands WAV) through input_in[1:0].
      3. Wait for kws_done on bidir_out[KWS_DONE_PAD].
      4. Read class index from bidir_out[5:3] and compare against ground truth.

    PDM driving mirrors test_full_pipeline_top.py exactly:
      pcm_to_pdm() in Python → one bit per chip clock → valid=input_in[1], data=input_in[0]
    """
    # Check data files
    manifest_path = _manifest_json_path()
    for p in [LOG_LUT_HEX, MEL_COEFF_HEX, MEL_INDEX_HEX,
              WEIGHTS_HEX, SCALES_TXT, manifest_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run src/ml/Pipeline/export.py then generate_spect_full.py first."
            )

    with open(manifest_path) as f:
        manifest = json.load(f)

    keyword     = manifest["keyword"]
    class_names = manifest["class_names"]
    samples     = manifest["samples"]
    dut._log.info(
        f"Manifest: {manifest_path}  keyword='{keyword}'  "
        f"({len(samples)} samples)"
    )

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await _do_reset(dut)

    # ---- Full UART boot ----
    dut._log.info("=== UART boot phase ===")
    await _boot_chip(dut, mini=False)

    # Use one manifest sample for the full-chip run. Override with
    # KWS_SAMPLE_INDEX or KWS_SAMPLE_MATCH when you want a different WAV.
    sample_idx, s = _select_manifest_sample(samples)
    gt_class = s["ground_truth_class"]
    gt_name  = s["ground_truth_name"]
    wav_path = s["wav"]
    dut._log.info(
        f"=== Audio phase: sample[{sample_idx}] {Path(wav_path).name}  GT={gt_name} ==="
    )

    # WAV → PCM → PDM
    pcm      = _load_wav_pcm(wav_path)
    pdm_bits = _pcm_to_pdm(pcm)
    dut._log.info(f"  {len(pcm)} PCM samples → {len(pdm_bits)} PDM bits")

    # Snapshot state right before the audio phase. The main wait loop below
    # drives PDM and monitors progress together to avoid two Python coroutines
    # waking on every clock edge.
    await RisingEdge(dut.clk)
    dut._log.info(f"  [initial state] {_probe_str(dut)}")

    # ---- Drive audio until KWS starts, then wait for KWS done ----
    # Budget: PDM stream + maximum inference time + CIC/FFT/LogMel pipeline latency
    kws_timeout_cycles = len(pdm_bits) + 20_000_000
    dut._log.info(f"  Waiting up to {kws_timeout_cycles:,} cycles for kws_done...")

    kws_done_seen = False
    rtl_class     = None

    LOG_EVERY = 500_000   # print a heartbeat every N sim cycles (~115s real at current speed)
    t0_real   = time.time()
    frontend_timeout_cycles = len(pdm_bits) + 200_000
    stage_handles = _resolve_probe_handles(dut)
    pulse_counts = {
        'fft_sync':        0,
        'filterbank_done': 0,
        'log_done':        0,
        'mel_valid':       0,
        'spect_done':      0,
        'kws_start':       0,
    }
    last_mel_cyc = None
    last_wr_addr = None
    last_wr_change_cyc = 0
    next_frame_log = 1
    MEL_STALL_WARN = 250_000
    kws_started_cyc = None
    KWS_LOG_EVERY = 100_000
    KWS_X_DEBUG = os.getenv("KWS_X_DEBUG", "1") != "0"
    KWS_X_FAIL = os.getenv("KWS_X_FAIL", "0") == "1"
    frontend_cyc = None
    milestones = {
        'cic_valid':        False,
        'fir_valid':        False,
        'fft_sync':         False,
        'fft_sync_aligned': False,
        'fft_valid':        False,
        'power_valid':      False,
        'filterbank_done':  False,
        'log_done':         False,
        'mel_valid':        False,
        'spect_done':       False,
        'kws_start':        False,
    }

    for cyc in range(frontend_timeout_cycles):
        if cyc < len(pdm_bits):
            dut.input_in.value = 0b10 | (pdm_bits[cyc] & 1)   # valid=1, data=bit
        else:
            dut.input_in.value = 0

        await RisingEdge(dut.clk)

        # Count recurring pulses, not just first-time milestones.
        for name in pulse_counts:
            if _handle_is_one(stage_handles[name]):
                pulse_counts[name] += 1

        for name in milestones:
            if not milestones[name] and _handle_is_one(stage_handles[name]):
                milestones[name] = True
                dut._log.info(f"  [milestone] {name} observed at cyc={cyc+1:,}")
                if name == 'kws_start':
                    kws_started_cyc = cyc + 1
                    dut._log.info(f"  [KWS start state] {_kws_probe_str(dut)}")

        if _handle_is_one(stage_handles['mel_valid']):
            wr_addr = _handle_int(stage_handles['spect_wr_addr'])
            mel_data = _handle_int(stage_handles['mel_data'])
            if wr_addr not in ('?', 'X') and wr_addr != last_wr_addr:
                last_wr_addr = wr_addr
                last_wr_change_cyc = cyc + 1

            last_mel_cyc = cyc + 1
            mel_count = pulse_counts['mel_valid']
            if mel_count <= 8:
                dut._log.info(
                    f"  [mel write] cyc={cyc+1:,} mel_count={mel_count} "
                    f"wr_addr={wr_addr} data={mel_data}"
                )
            if mel_count % 40 == 0:
                frame = mel_count // 40
                if frame == next_frame_log or frame % 10 == 0:
                    dut._log.info(
                        f"  [spect frame] frame={frame}/50 "
                        f"mel_count={mel_count}/2000 wr_addr={wr_addr}"
                    )
                    next_frame_log = frame + 1

        if (
            milestones['mel_valid']
            and not milestones['spect_done']
            and last_mel_cyc is not None
            and (cyc + 1 - last_mel_cyc) == MEL_STALL_WARN
        ):
            wr_addr = _handle_int(stage_handles['spect_wr_addr'])
            dut._log.warning(
                f"  [stall?] no mel_valid for {MEL_STALL_WARN:,} cycles; "
                f"counts={pulse_counts} wr_addr={wr_addr} "
                f"last_wr_change_cyc={last_wr_change_cyc:,} probes={_probe_str(dut)}"
            )

        done, cls = _read_kws_pads(dut)
        if done:
            rtl_class     = cls
            kws_done_seen = True
            frontend_cyc  = cyc + 1
            break
        if kws_started_cyc is not None:
            frontend_cyc = cyc + 1
            break

        if cyc + 1 == frontend_timeout_cycles and not milestones['spect_done']:
            missing = _missing_milestone(milestones)
            raise AssertionError(
                f"frontend never produced spect_done within {frontend_timeout_cycles:,} cycles; "
                f"first missing milestone: {missing}; counts={pulse_counts}; "
                f"last_wr_addr={last_wr_addr}; last_wr_change_cyc={last_wr_change_cyc:,}; "
                f"probes: {_probe_str(dut)}"
            )
        if cyc % LOG_EVERY == LOG_EVERY - 1:
            elapsed = time.time() - t0_real
            sim_ns  = (cyc + 1) * CLK_PERIOD_NS
            pdm_status = 'done' if cyc + 1 >= len(pdm_bits) else f'in flight ({cyc+1}/{len(pdm_bits)})'
            dut._log.info(
                f"  [KWS heartbeat] cyc={cyc+1:,}  sim={sim_ns/1e6:.1f}ms  "
                f"real={elapsed:.0f}s  PDM={pdm_status}"
            )
            dut._log.info(f"  [KWS internals] {_probe_str(dut)}")
            dut._log.info(
                f"  [frontend counts] {pulse_counts} "
                f"last_mel_cyc={last_mel_cyc} last_wr_change_cyc={last_wr_change_cyc}"
            )

    if frontend_cyc is None:
        frontend_cyc = frontend_timeout_cycles

    if not kws_done_seen and kws_started_cyc is None:
        missing = _missing_milestone(milestones)
        raise AssertionError(
            f"frontend never produced kws_start within {frontend_timeout_cycles:,} cycles; "
            f"first missing milestone: {missing}; counts={pulse_counts}; "
            f"last_wr_addr={last_wr_addr}; last_wr_change_cyc={last_wr_change_cyc:,}; "
            f"probes: {_probe_str(dut)}"
        )

    if not kws_done_seen:
        remaining_pdm = pdm_bits[frontend_cyc:]
        if remaining_pdm:
            cocotb.start_soon(_drive_pdm(dut, remaining_pdm))
            dut._log.info(f"  Continuing remaining PDM in background ({len(remaining_pdm):,} bits)")
        else:
            dut.input_in.value = 0

        done_handle = stage_handles['kws_done']
        monitor_task = cocotb.start_soon(
            _monitor_kws_progress(dut, kws_started_cyc, done_handle, KWS_LOG_EVERY)
        )
        x_monitor_task = None
        if KWS_X_DEBUG:
            x_monitor_task = cocotb.start_soon(
                _monitor_kws_x_sources(
                    dut,
                    done_handle,
                    log_limit=int(os.getenv("KWS_X_LOG_LIMIT", "24")),
                    fail_fast=KWS_X_FAIL,
                )
            )
            dut._log.info(
                f"  KWS X-source monitor enabled "
                f"(fail_fast={KWS_X_FAIL}, log_limit={os.getenv('KWS_X_LOG_LIMIT', '24')})"
            )
        remaining_timeout_cycles = max(1, kws_timeout_cycles - frontend_cyc)
        dut._log.info(
            f"  KWS started; waiting on done edge for up to "
            f"{remaining_timeout_cycles:,} more cycles..."
        )
        result = await First(
            RisingEdge(done_handle),
            Timer(remaining_timeout_cycles * CLK_PERIOD_NS, units="ns"),
        )
        if hasattr(monitor_task, "kill"):
            monitor_task.kill()
        if x_monitor_task is not None and hasattr(x_monitor_task, "kill"):
            x_monitor_task.kill()
        if isinstance(result, Timer):
            raise AssertionError(
                f"kws_done never asserted within {kws_timeout_cycles:,} total cycles "
                f"(frontend_cyc={frontend_cyc:,}; real time: {time.time()-t0_real:.0f}s); "
                f"KWS: {_kws_probe_str(dut)}"
            )
        await Timer(1, units="step")
        done, cls = _read_kws_pads(dut)
        rtl_class = cls
        kws_done_seen = bool(done)

    assert kws_done_seen, \
        f"kws_done never asserted within {kws_timeout_cycles:,} cycles  " \
        f"(real time: {time.time()-t0_real:.0f}s)"

    rtl_name = class_names[rtl_class] if rtl_class < len(class_names) else f"cls{rtl_class}"
    passed   = (rtl_class == gt_class)

    dut._log.info(f"  RTL class = {rtl_name} ({rtl_class})  |  GT = {gt_name} ({gt_class})")
    dut._log.info(f"  {'PASS' if passed else 'FAIL'}")

    assert passed, \
        f"KWS misclassified '{gt_name}' as '{rtl_name}' — " \
        f"check chip_core wiring or re-run generate_spect_full.py"

    dut._log.info("test_chip_core_e2e PASSED")
