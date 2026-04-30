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

import json, sys, time
import cocotb
from cocotb.clock    import Clock
from cocotb.triggers import RisingEdge, ClockCycles, First, Timer
from pathlib         import Path

# ---------------------------------------------------------------------------
# Slot parameters (1×1)
# ---------------------------------------------------------------------------
NUM_INPUT_PADS = 12
NUM_BIDIR_PADS = 40

# ---------------------------------------------------------------------------
# Pad indices — must match chip_core.sv localparam
# ---------------------------------------------------------------------------
UART_RX_PAD    = 0
UART_TX_PAD    = 1
KWS_DONE_PAD   = 2
KWS_CLASS_BASE = 3   # class[0]=pad3, class[1]=pad4, class[2]=pad5

PDM_DATA_PAD   = 0   # input_in[0]
PDM_VALID_PAD  = 1   # input_in[1]

# ---------------------------------------------------------------------------
# Clock / UART timing
#   Chip clock  = 25 MHz  → CLK_PERIOD_NS = 40 ns
#   RTL uses prescale=1 under `ifdef SIM → 8 cycles/bit (27× faster than real)
#   Real UART prescale=27 → 216 cycles/bit at 115200 baud
# ---------------------------------------------------------------------------
CLK_PERIOD_NS = 40
UART_PRESCALE = 1    # matches `ifdef SIM in chip_core.sv
BIT_CYCLES    = UART_PRESCALE * 8   # 8 cycles/bit in sim

# ---------------------------------------------------------------------------
# Boot protocol constants — must match boot_pkg.sv
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_SRC       = Path(__file__).resolve().parent.parent / "src"
_ML        = _SRC / "ml"
_RTL       = _SRC / "rtl"
_KWS_DIR   = _RTL / "dscnn/kws_top"
_LOGMEL    = _RTL / "Log-Mel/data"
_MODEL_DIR = _ML / "models/dscnn-32requant-v11"

LOG_LUT_HEX   = _LOGMEL    / "log2_lut.hex"
MEL_COEFF_HEX = _LOGMEL    / "mel_coeffs_sparse.hex"
MEL_INDEX_HEX = _LOGMEL    / "mel_indices.hex"
WEIGHTS_HEX   = _MODEL_DIR / "weights.hex"
SCALES_TXT    = _MODEL_DIR / "scales.txt"
MANIFEST_JSON = _KWS_DIR   / "test_vectors.json"
SPECT_DIR     = _KWS_DIR   / "spectrograms"

# ---------------------------------------------------------------------------
# Hex file loader
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Layer-config packing
#   Replicates program_layers() from test_kws_top.py but produces flat byte
#   arrays suitable for UART packet payloads.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# UART helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------------------------

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
        layer_cfgs   = load_layer_cfgs(SCALES_TXT)
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

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

async def _do_reset(dut):
    dut.rst_n.value    = 0
    dut.bidir_in.value = 1   # UART RX idle (high)
    dut.input_in.value = 0
    await ClockCycles(dut.clk, 20)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)

# ---------------------------------------------------------------------------
# PDM audio helpers  (mirrors pcm_to_pdm / drive_pdm in test_full_pipeline_top.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# KWS output reader
# ---------------------------------------------------------------------------

def _read_kws_pads(dut):
    def _safe(idx):
        b = _bidir_bit(dut, idx)
        return b if b is not None else 0   # X → 0 (not done / class 0)
    kws_done  = _safe(KWS_DONE_PAD)
    kws_class = (_safe(KWS_CLASS_BASE+2) << 2) | (_safe(KWS_CLASS_BASE+1) << 1) | _safe(KWS_CLASS_BASE)
    return kws_done, kws_class

# ---------------------------------------------------------------------------
# Internal-signal probe for debugging
# ---------------------------------------------------------------------------

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
        # KWS FSM state (best-effort: hierarchy depends on kws_top internals)
        'fsm_state':        _get('kws_inst.u_fsm.state'),
        # spect_buffer write-select (tells us which spectrogram bank is active)
        'spect_write_sel':  _get('spect_write_sel'),
    }


def _probe_str(dut):
    p = _probe(dut)
    return (
        f"boot_done={p['boot_done']}  inf_rst={p['inference_reset']}  "
        f"pdm_valid={p['pdm_valid']}  spect_done={p['spect_done']}  "
        f"spect_sel={p['spect_write_sel']}  kws_start={p['kws_start']}  "
        f"kws_done={p['kws_done']}  fsm={p['fsm_state']}"
    )


# ---------------------------------------------------------------------------
# Test 1 — boot protocol smoke-test
# ---------------------------------------------------------------------------

# @cocotb.test()
# async def test_chip_core_boot(dut):
#     """
#     Mini UART boot with synthetic data.
#     Checks:
#       • every boot packet receives ACK (no NACK / timeout)
#       • pad output-enable directions after reset:
#           UART_RX  = input  (OE=0, IE=1)
#           UART_TX  = output (OE=1, IE=0)
#           KWS pads = output (OE=1)
#     """
#     cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
#     await _do_reset(dut)

#     # --- Pad direction checks ---
#     oe = int(dut.bidir_oe.value)
#     ie = int(dut.bidir_ie.value)

#     assert not ((oe >> UART_RX_PAD) & 1),   "UART_RX_PAD OE must be 0 (input)"
#     assert     ((ie >> UART_RX_PAD) & 1),   "UART_RX_PAD IE must be 1"
#     assert     ((oe >> UART_TX_PAD) & 1),   "UART_TX_PAD OE must be 1 (output)"
#     assert not ((ie >> UART_TX_PAD) & 1),   "UART_TX_PAD IE must be 0"
#     for pad in (KWS_DONE_PAD, KWS_CLASS_BASE, KWS_CLASS_BASE+1, KWS_CLASS_BASE+2):
#         assert (oe >> pad) & 1, f"KWS pad {pad} OE must be 1 (output)"

#     dut._log.info("Pad directions OK")

#     # --- Mini boot ---
#     dut._log.info("Starting mini boot sequence...")
#     await _boot_chip(dut, mini=True)
#     dut._log.info("test_chip_core_boot PASSED")


# ---------------------------------------------------------------------------
# Test 2 — full end-to-end KWS
# ---------------------------------------------------------------------------

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
    for p in [LOG_LUT_HEX, MEL_COEFF_HEX, MEL_INDEX_HEX,
              WEIGHTS_HEX, SCALES_TXT, MANIFEST_JSON]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run src/ml/Pipeline/export.py then generate_spect_full.py first."
            )

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    keyword     = manifest["keyword"]
    class_names = manifest["class_names"]
    samples     = manifest["samples"]
    dut._log.info(f"Keyword: '{keyword}'  ({len(samples)} samples in manifest)")

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await _do_reset(dut)

    # ---- Full UART boot ----
    dut._log.info("=== UART boot phase ===")
    await _boot_chip(dut, mini=False)

    # Use first sample (keep sim time reasonable; module-level tests cover full sets)
    s        = samples[0]
    gt_class = s["ground_truth_class"]
    gt_name  = s["ground_truth_name"]
    wav_path = s["wav"]
    dut._log.info(f"=== Audio phase: {Path(wav_path).name}  GT={gt_name} ===")

    # WAV → PCM → PDM
    pcm      = _load_wav_pcm(wav_path)
    pdm_bits = _pcm_to_pdm(pcm)
    dut._log.info(f"  {len(pcm)} PCM samples → {len(pdm_bits)} PDM bits")

    # Drive PDM asynchronously while polling for kws_done
    cocotb.start_soon(_drive_pdm(dut, pdm_bits))

    # ---- Wait for kws_done ----
    # Budget: PDM stream + maximum inference time + CIC/FFT/LogMel pipeline latency
    kws_timeout_cycles = len(pdm_bits) + 20_000_000
    dut._log.info(f"  Waiting up to {kws_timeout_cycles:,} cycles for kws_done...")

    kws_done_seen = False
    rtl_class     = None

    LOG_EVERY = 1_000_000   # print a heartbeat every N sim cycles
    t0_real   = time.time()

    for cyc in range(kws_timeout_cycles):
        await RisingEdge(dut.clk)
        done, cls = _read_kws_pads(dut)
        if done:
            rtl_class     = cls
            kws_done_seen = True
            break
        if cyc % LOG_EVERY == LOG_EVERY - 1:
            elapsed = time.time() - t0_real
            sim_ns  = (cyc + 1) * CLK_PERIOD_NS
            dut._log.info(
                f"  [KWS] {cyc+1:,} cycles ({sim_ns/1e6:.1f} ms sim)  "
                f"real {elapsed:.0f}s  —  PDM stream {'done' if cyc+1 >= len(pdm_bits) else 'in flight'}"
            )

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

