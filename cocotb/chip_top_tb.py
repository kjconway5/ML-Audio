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
gl = os.getenv("GL", "0").lower() in ("1", "true", "yes")
sdf_corner = os.getenv("SDF_CORNER", "")
slot = os.getenv("SLOT", "1x1")

hdl_toplevel = "chip_top"

def _bidir_drive_value(uart_rx_bit=1):
    # str(LogicArray) is MSB-first, while pad bit numbers are LSB-first.
    bits = ["Z"] * NUM_BIDIR_PADS
    bits[UART_RX_PAD] = "1" if uart_rx_bit else "0"
    bits[AUDIO_TEST_MODE_PAD] = "0"
    bits[ML_TEST_MODE_PAD] = "0"
    return "".join(reversed(bits))


async def set_defaults(dut):
    dut.input_PAD.value = 0
    # UART RX is held idle-high. The test-mode bidir pads are external inputs,
    # so drive them low instead of leaving the SRAM debug mux select floating.
    dut.bidir_PAD.value = _bidir_drive_value(1)

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
    await reset(dut.rst_n_PAD, time_ns=4000 if gl else 1000)


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
AUDIO_TEST_MODE_PAD = 7
ML_TEST_MODE_PAD = 8
CLK_PERIOD_NS = 40
UART_PRESCALE = 17 if gl else 1
BIT_CYCLES = UART_PRESCALE * 8
SYNC_0 = 0xAA
SYNC_1 = 0x55
ACK_BYTE = 0x06
SCORE_HEADER = 0xDA   # header byte of the 29-byte score TX packet
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
    # Drive only external input pads: UART RX plus test-mode selects.
    # Other bidirectional pads remain Z so chip_top can drive TX/KWS outputs.
    dut.bidir_PAD.value = _bidir_drive_value(bit)


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


def _gl_probe_chip_core_signals(dut, log):
    """Probe preserved chip_core nets in the flat PNL to debug score TX startup.

    Net names verified against final/pnl/chip_top.pnl.v.
    score_tx_active/kws_done_r/uart_txd were all renamed to anonymous wires
    by synthesis, so we probe the uart_tx sub-module registers instead.
    Never raises.
    """
    _GL_PROBES = [
        r"\i_chip_core.u_uart_tx.txd_reg ",           # UART TX output bit (1=idle)
        r"\i_chip_core.u_uart_tx.s_axis_tready_reg ",  # tx_ready (1=accepting data)
        r"\i_chip_core.u_uart_tx.bit_cnt[3] ",         # bit counter bit-3
        r"\i_chip_core.u_uart_tx.bit_cnt[0] ",         # bit counter bit-0
        r"\i_chip_core.kws_inst.inst_ctrl.state[0] ",  # FSM state bit-0
        r"\i_chip_core.kws_inst.inst_ctrl.state[1] ",  # FSM state bit-1
    ]
    results = []
    for name in _GL_PROBES:
        try:
            val = getattr(dut.u_chip_top, name).value
            results.append(f"{name.strip()}={val}")
        except Exception:
            results.append(f"{name.strip()}=N/A")
    try:
        log.info(f"  [gl_probe] chip_core signals: {' | '.join(results)}")
    except Exception:
        pass


def _gl_read_gap_scores_pnl(dut, log):
    """Read global_pool_acc[0..6][31:0] from the PNL wire-by-wire.

    These names are preserved in final/pnl/chip_top.pnl.v.
    Returns list of (class_idx, int32_value) or None on failure.
    """
    scores = []
    for c in range(7):
        raw = 0
        ok = True
        for b in range(32):
            name = rf"\i_chip_core.kws_inst.inst_ctrl.global_pool_acc[{c}][{b}] "
            try:
                bit = int(getattr(dut.u_chip_top, name).value)
                raw |= (bit << b)
            except Exception:
                ok = False
                break
        if not ok:
            log.warning(f"  [gl_gap] global_pool_acc[{c}] not accessible")
            return None
        if raw >= (1 << 31):
            raw -= (1 << 32)
        scores.append((c, raw))
    return scores


_GL_MILESTONE_BITS = {
    'boot_done': r"\i_chip_core.boot_done ",
    'spect_write_sel': r"\i_chip_core.kws_inst.inst_ctrl.spect_write_sel ",
    'spect_done': r"\i_chip_core.kws_inst.inst_ctrl.spect_done ",
    'kws_done_r': r"\i_chip_core.kws_done_r ",
}

_GL_MILESTONE_BUSES = {
    'frame_state': (r"\i_chip_core.frame_control_state[{}] ", 2),
    'kws_state': (r"\i_chip_core.kws_inst.inst_ctrl.state[{}] ", 4),
    'kws_layer': (r"\i_chip_core.kws_inst.inst_ctrl.layer[{}] ", 4),
}


def _gl_read_escaped(chip_top, name):
    """Read one escaped-name bit from the flat PNL: int, 'X', or None if absent."""
    try:
        handle = getattr(chip_top, name)
    except Exception:
        return None
    try:
        return int(handle.value)
    except Exception:
        return 'X'


def _gl_milestone_str(dut):
    """Compact frontend/KWS progress summary from nets preserved in the PNL.

    Net names verified against final/pnl/chip_top.pnl.v. Usable on every
    heartbeat in GL/SDF mode where the RTL probe hierarchy is unavailable,
    so a hang localizes to a stage instead of a silent pad wait. Never raises.
    """
    chip_top = getattr(dut, "u_chip_top", None)
    if chip_top is None:
        return "u_chip_top unavailable"
    parts = []
    for label, name in _GL_MILESTONE_BITS.items():
        val = _gl_read_escaped(chip_top, name)
        parts.append(f"{label}={'?' if val is None else val}")
    for label, (fmt, width) in _GL_MILESTONE_BUSES.items():
        bits = [_gl_read_escaped(chip_top, fmt.format(b)) for b in range(width)]
        if any(b is None for b in bits):
            parts.append(f"{label}=?")
        elif any(b == 'X' for b in bits):
            parts.append(f"{label}=X")
        else:
            parts.append(f"{label}={sum(b << i for i, b in enumerate(bits))}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# GL X-propagation forensics
#
# The KWS layer controller (inst_ctrl / FSM.sv) state goes X in gate-level sim
# the instant it activates after the first spectrogram (see sim_gl.log). These
# helpers + the _GLForensics tracker capture *which* register corrupts first and
# what the surrounding datapath looks like at that exact cycle, so an overnight
# run pinpoints the X source without a second pass. All names verified to
# survive synthesis in final/pnl/chip_top.pnl.v.
# ---------------------------------------------------------------------------

_CTRL = r"\i_chip_core.kws_inst.inst_ctrl."

# Control / FSM regs that are RESET to a known value (so a transition to X is a
# real corruption event, not a power-up X). Sampled every cycle in the window.
# (label, name-format, width) — width None => scalar (no bit index).
_GL_FSM_TRIGGER = [
    ('state',           _CTRL + r"state[{}] ", 4),
    ('layer',           _CTRL + r"layer[{}] ", 4),
    ('ic',              _CTRL + r"ic[{}] ", 8),
    ('kh',              _CTRL + r"kh[{}] ", 4),
    ('kw',              _CTRL + r"kw[{}] ", 4),
    ('oc',              _CTRL + r"oc[{}] ", 8),
    ('oh',              _CTRL + r"oh[{}] ", 8),
    ('ow',              _CTRL + r"ow[{}] ", 8),
    ('spect_write_sel', _CTRL + r"spect_write_sel ", None),
    ('spect_read_sel',  _CTRL + r"spect_read_sel ", None),
    ('spect_ready',     _CTRL + r"spect_ready ", None),
    ('spect_done',      _CTRL + r"spect_done ", None),
    ('start',           _CTRL + r"start ", None),
    ('mac_en',          _CTRL + r"mac_en ", None),
    ('mac_clear',       _CTRL + r"mac_clear ", None),
    ('buf_sel',         _CTRL + r"buf_sel ", None),
    ('cfg_load_done',   _CTRL + r"cfg_load_done ", None),
]

# Wider datapath regs — read only on events (first-spect, first-X, burst,
# periodic, end) to keep per-cycle overhead low.
_GL_FSM_DATAPATH = [
    ('mac_acc',         _CTRL + r"mac_acc[{}] ", 32),
    ('mac_ifmap',       _CTRL + r"mac_ifmap[{}] ", 8),
    ('mac_weight',      _CTRL + r"mac_weight[{}] ", 8),
    ('mac_bias',        _CTRL + r"mac_bias[{}] ", 32),
    ('max_val',         _CTRL + r"max_val[{}] ", 32),
    ('max_idx',         _CTRL + r"max_idx[{}] ", 3),
    ('global_pool_idx', _CTRL + r"global_pool_idx[{}] ", 3),
    ('rq_shift',        _CTRL + r"rq_shift[{}] ", 5),
    ('rq_mult',         _CTRL + r"rq_mult[{}] ", 32),
    ('rq_relu_en',      _CTRL + r"rq_relu_en ", None),
]


def _fmt_val(v):
    return '?' if v is None else ('X' if v == 'X' else str(v))


def _sim_ms(cyc):
    return f"{cyc * CLK_PERIOD_NS / 1e6:.3f}ms"


class _GLForensics:
    """Per-cycle X-corruption tracker for the gate-level KWS controller.

    Tunables (env):
      KWS_ARM_BEFORE      cycles before PDM-end to start sampling (default 80000)
      KWS_TRIGGER_WINDOW  cycles to keep per-cycle sampling armed   (default 1500000)
      KWS_XTRACE          cycles of per-cycle full dump after first X (default 256)
      KWS_FINE_EVERY      periodic full-snapshot interval in window (default 20000)
    """

    def __init__(self, dut, log, pdm_len):
        self.dut = dut
        self.log = log
        self.chip_top = getattr(dut, "u_chip_top", None)
        self.arm_cyc = max(0, pdm_len - int(os.getenv("KWS_ARM_BEFORE", "80000")))
        self.window = int(os.getenv("KWS_TRIGGER_WINDOW", "1500000"))
        self.xtrace = int(os.getenv("KWS_XTRACE", "256"))
        self.fine_every = int(os.getenv("KWS_FINE_EVERY", "20000"))
        self._handles = {}
        self.prev = {}
        self.firstx = []          # ordered (label, cyc, prev_value) corruption events
        self.firstx_set = set()
        self.burst_remaining = 0
        self.burst_fired = False
        self.spect_flip_logged = False
        self.spect_done_logged = False
        self.last_full_cyc = -10 ** 9
        if self.chip_top is None:
            self.log.warning("  [GL-FORENSIC] u_chip_top unavailable; forensics disabled")

    # -- cached low-level reads -------------------------------------------
    def _handle(self, name):
        h = self._handles.get(name, 0)
        if h == 0:
            try:
                h = getattr(self.chip_top, name)
            except Exception:
                h = None
            self._handles[name] = h
        return h

    def _scalar(self, name):
        h = self._handle(name)
        if h is None:
            return None
        try:
            return int(h.value)
        except Exception:
            return 'X'

    def _bus(self, fmt, width, signed=False):
        bits = [self._scalar(fmt.format(b)) for b in range(width)]
        if all(b is None for b in bits):
            return None
        if any(b == 'X' or b is None for b in bits):
            return 'X'
        v = sum((b & 1) << i for i, b in enumerate(bits))
        if signed and v >= (1 << (width - 1)):
            v -= (1 << width)
        return v

    def _snapshot(self, table):
        snap = {}
        for label, fmt, width in table:
            snap[label] = self._scalar(fmt) if width is None else self._bus(fmt, width)
        return snap

    @staticmethod
    def _snap_str(snap):
        return ' '.join(f"{k}={_fmt_val(v)}" for k, v in snap.items())

    # -- forensic dumps ----------------------------------------------------
    def _full_dump(self, cyc, tag):
        t = self._snapshot(_GL_FSM_TRIGGER)
        d = self._snapshot(_GL_FSM_DATAPATH)
        self.log.info(f"  [GL-SNAP {tag} cyc={cyc:,} sim={_sim_ms(cyc)}] "
                      f"{self._snap_str(t)} || {self._snap_str(d)}")

    def _spect_dump(self, cyc):
        try:
            _, summ = _sample_spect_sram_gl(self.dut, n_samples=24)
            self.log.info(f"  [GL-SPECT cyc={cyc:,}] {summ}")
        except Exception as e:
            self.log.info(f"  [GL-SPECT cyc={cyc:,}] sample failed: {type(e).__name__}: {e}")

    def _gap_dump(self, cyc):
        parts = []
        for c in range(7):
            v = self._bus(_CTRL + rf"global_pool_acc[{c}][{{}}] ", 32, signed=True)
            parts.append(f"cls{c}={_fmt_val(v)}")
        self.log.info(f"  [GL-GAP cyc={cyc:,}] " + " ".join(parts))

    # -- per-cycle entry point --------------------------------------------
    def tick(self, cyc):
        """Call once per clock with the 1-based cycle count (cyc+1 from loop)."""
        if self.chip_top is None:
            return

        # First spectrogram completion: frontend flipped the write bank. Capture
        # the spectrogram content now to prove whether the frontend wrote valid
        # data (vs X) BEFORE the KWS engine reads it.
        if not self.spect_flip_logged:
            sw = self._scalar(_CTRL + r"spect_write_sel ")
            if sw not in (None, 0, 'X'):
                self.spect_flip_logged = True
                self.log.info(f"  [GL-EVENT] spect_write_sel->{sw} (first spectrogram written) "
                              f"cyc={cyc:,} sim={_sim_ms(cyc)}")
                self._full_dump(cyc, "first-spect-write")
                self._spect_dump(cyc)

        if not self.spect_done_logged:
            if self._scalar(_CTRL + r"spect_done ") == 1:
                self.spect_done_logged = True
                self.log.info(f"  [GL-EVENT] spect_done pulse cyc={cyc:,} sim={_sim_ms(cyc)}")
                self._full_dump(cyc, "spect_done")

        armed = self.arm_cyc < cyc <= self.arm_cyc + self.window
        in_burst = self.burst_remaining > 0
        if not armed and not in_burst:
            return

        if armed and (cyc - self.last_full_cyc) >= self.fine_every:
            self._full_dump(cyc, "periodic")
            self.last_full_cyc = cyc

        snap = self._snapshot(_GL_FSM_TRIGGER)

        # A reset reg going from a defined value to X is a real corruption event.
        # (Unreset datapath regs are born X and are reported via the burst dump,
        # not here, to avoid power-up-X noise.)
        newly = [(label, self.prev.get(label)) for label, val in snap.items()
                 if val == 'X' and label not in self.firstx_set
                 and isinstance(self.prev.get(label), int)]
        for label, prevv in newly:
            self.firstx_set.add(label)
            self.firstx.append((label, cyc, prevv))
            self.log.warning(f"  [GL-FIRST-X] {label} {_fmt_val(prevv)}->X "
                             f"cyc={cyc:,} sim={_sim_ms(cyc)}")
        if newly and not self.burst_fired:
            self.burst_fired = True
            self.burst_remaining = self.xtrace
            self.log.warning(f"  [GL-FIRST-X] first corruption at cyc={cyc:,}; "
                             f"forensic dump + {self.xtrace}-cycle trace follows")
            self._full_dump(cyc, "first-X")
            self._spect_dump(cyc)
            self._gap_dump(cyc)

        if in_burst:
            self.log.info(f"  [GL-XTRACE cyc={cyc:,}] {self._snap_str(snap)}")
            self.burst_remaining -= 1
            if self.burst_remaining == 0:
                self._full_dump(cyc, "burst-end")
                self._gap_dump(cyc)

        self.prev = snap

    def summary(self, cyc):
        if self.chip_top is None:
            return
        self.log.warning("  [GL-FORENSIC] ===== end-of-run X summary =====")
        if self.firstx:
            self.log.warning("  [GL-FORENSIC] reset-reg corruption order (label @ cyc, prev->X):")
            for label, c, prevv in self.firstx:
                self.log.warning(f"      {label:<16} @ cyc={c:,} sim={_sim_ms(c)} ({_fmt_val(prevv)}->X)")
        else:
            self.log.warning("  [GL-FORENSIC] no reset-reg corruption captured in window "
                             "(widen KWS_TRIGGER_WINDOW / KWS_ARM_BEFORE)")
        self._full_dump(cyc, "final")
        self._spect_dump(cyc)
        self._gap_dump(cyc)


async def _uart_read_byte(dut, timeout_cycles=800_000):
    log = dut._log
    log.info("  [UART RX] waiting for start bit on bidir_PAD[1] (chip TX)")
    log_interval = 5_000 if gl else 50_000
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk_PAD)
        if _uart_tx_pad_bit(dut) == 0:
            log.info(f"  [UART RX] start bit detected at poll cycle {cyc}")
            break
        if cyc % log_interval == log_interval - 1:
            try:
                bidir_str = str(dut.bidir_PAD.value)
            except Exception:
                bidir_str = "?"
            log.info(f"  [UART RX] still idle at poll cycle {cyc}, TX={_uart_tx_pad_bit(dut)} bidir={bidir_str}")
            if gl and cyc < log_interval * 3:
                _gl_probe_chip_core_signals(dut, log)
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

    candidate = _KWS_DIR / path
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

    if sample_index:
        idx = int(sample_index, 0)
        if idx < 0 or idx >= len(samples):
            raise AssertionError(f"KWS_SAMPLE_INDEX={idx} is out of range; manifest has {len(samples)} samples")
        if sample_keyword and (idx, samples[idx]) not in filtered:
            raise AssertionError(
                f"KWS_SAMPLE_INDEX={idx} exists, but its ground_truth_name is "
                f"{samples[idx].get('ground_truth_name')!r}, not KWS_KEYWORD={sample_keyword!r}"
            )
        return idx, samples[idx]

    if sample_match:
        needle = sample_match.lower()
        for idx, sample in filtered:
            if needle in str(sample.get("wav", "")).lower() or needle in str(sample.get("ground_truth_name", "")).lower():
                return idx, sample
        raise AssertionError(f"KWS_SAMPLE_MATCH={sample_match!r} did not match any selected manifest sample")

    return filtered[0]


def _core(dut):
    if gl:
        try:
            return dut.u_chip_top.i_chip_core
        except AttributeError:
            return None  # PnR flattened i_chip_core into chip_top — probes unavailable
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
        # The STFFT was rewritten to an AXI-stream interface: it no longer has
        # an o_fft_sync port, and full_pipeline_top renamed fft_sync_aligned to
        # the registered fft_sync_r/fft_sync_rr pair (rr is the logmel-aligned one).
        'fft_sync': 'pipeline_inst.fft_sync_r',
        'fft_sync_aligned': 'pipeline_inst.fft_sync_rr',
        'fft_valid': 'pipeline_inst.fft_valid',
        # STFFT internals — let a frontend hang localize to a stage instead of a
        # silent "fft never produced anything". warmup_cnt climbing to 255 means
        # samples are reaching the FFT; if it stalls below 255 the stall is
        # upstream (CIC/FIR feed), if it completes but no o_valid appears the
        # stall is the FFT-core handshake (watch fft_done / r2fft_s_ready).
        'stfft_warmup_cnt': 'pipeline_inst.u_stfft.warmup_cnt',
        'stfft_warmup_done': 'pipeline_inst.u_stfft.warmup_done',
        'stfft_frame_pending': 'pipeline_inst.u_stfft.frame_pending',
        'stfft_state': 'pipeline_inst.u_stfft.state',
        'stfft_read_idx': 'pipeline_inst.u_stfft.read_idx',
        'stfft_i_ready': 'pipeline_inst.stfft_i_ready',
        'stfft_fft_i_ready': 'pipeline_inst.u_stfft.fft_i_ready_w',
        'fft_done': 'pipeline_inst.u_stfft.u_fft.done',
        'r2fft_s_ready': 'pipeline_inst.u_stfft.u_fft.r2fft_s_ready',
        'power_valid': 'pipeline_inst.u_logmel.power_valid',
        'fb_state': 'pipeline_inst.u_logmel.u_mel_filterbank.state',
        'fb_store_ctr': 'pipeline_inst.u_logmel.u_mel_filterbank.store_ctr',
        'fb_mel_idx': 'pipeline_inst.u_logmel.u_mel_filterbank.mel_idx',
        'fb_proc_bin': 'pipeline_inst.u_logmel.u_mel_filterbank.proc_bin',
        'fb_start_bin': 'pipeline_inst.u_logmel.u_mel_filterbank.start_bin_r',
        'fb_end_bin': 'pipeline_inst.u_logmel.u_mel_filterbank.end_bin_r',
        'fb_coeff_base': 'pipeline_inst.u_logmel.u_mel_filterbank.coeff_base',
        'fb_index_addr': 'pipeline_inst.u_logmel.u_mel_filterbank.index_addr',
        'fb_index_out': 'pipeline_inst.u_logmel.u_mel_filterbank.index_out',
        'fb_coeff_addr': 'pipeline_inst.u_logmel.u_mel_filterbank.coeff_addr',
        'fb_weight': 'pipeline_inst.u_logmel.u_mel_filterbank.weight',
        'frame_ctrl_state': 'pipeline_inst.u_logmel.u_frame_ctrl.curr_state_q',
        'vad_active': 'pipeline_inst.u_logmel.vad_active',
        'vad_done': 'pipeline_inst.u_logmel.vad_done',
        'vad_frame_accept': 'pipeline_inst.u_logmel.vad_frame_accept',
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
    if root is None:
        return {name: None for name in paths}
    for name, path in paths.items():
        try:
            handles[name] = _get_path(root, path)
        except Exception:
            handles[name] = None
    return handles


def _frontend_probes_available(handles):
    """Return true when the non-pad frontend milestones are observable.

    Gate-level/PnR netlists may flatten or rename chip_core internals. In that
    case the pad-level test must not fail just because debug probes disappeared.
    """
    required = (
        "cic_valid", "fir_valid", "fft_sync", "fft_sync_aligned", "fft_valid",
        "power_valid", "filterbank_done", "log_done", "mel_valid",
        "spect_done", "kws_start",
    )
    return all(handles.get(name) is not None for name in required)


def _probe_str(dut):
    handles = _resolve_probe_handles(dut)
    return (
        f"cic={_handle_int(handles['cic_valid'])} fir={_handle_int(handles['fir_valid'])} "
        f"fft_sync={_handle_int(handles['fft_sync'])}/{_handle_int(handles['fft_sync_aligned'])} "
        f"fft_valid={_handle_int(handles['fft_valid'])} pwr={_handle_int(handles['power_valid'])} "
        f"stfft[warm={_handle_int(handles['stfft_warmup_cnt'])} done={_handle_int(handles['stfft_warmup_done'])} "
        f"pend={_handle_int(handles['stfft_frame_pending'])} state={_handle_int(handles['stfft_state'])} "
        f"rd={_handle_int(handles['stfft_read_idx'])} iready={_handle_int(handles['stfft_i_ready'])} "
        f"fft_iready={_handle_int(handles['stfft_fft_i_ready'])} fft_done={_handle_int(handles['fft_done'])} "
        f"r2ready={_handle_int(handles['r2fft_s_ready'])}] "
        f"fb[state={_handle_int(handles['fb_state'])} store={_handle_int(handles['fb_store_ctr'])} mel={_handle_int(handles['fb_mel_idx'])} "
        f"proc={_handle_int(handles['fb_proc_bin'])} start={_handle_int(handles['fb_start_bin'])} end={_handle_int(handles['fb_end_bin'])} "
        f"base={_handle_int(handles['fb_coeff_base'])} iaddr={_handle_int(handles['fb_index_addr'])} iout={_handle_int(handles['fb_index_out'])} "
        f"caddr={_handle_int(handles['fb_coeff_addr'])} weight={_handle_int(handles['fb_weight'])}] "
        f"vad[active={_handle_int(handles['vad_active'])} done={_handle_int(handles['vad_done'])} accept={_handle_int(handles['vad_frame_accept'])}] "
        f"fc={_handle_int(handles['frame_ctrl_state'])} fb_done={_handle_int(handles['filterbank_done'])} log_done={_handle_int(handles['log_done'])} "
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


async def _uart_read_score_packet(dut, class_names, timeout_cycles=500_000):
    """Read the 29-byte score packet sent by chip_core after kws_done.

    Packet format (chip_core.sv score TX sequencer):
      byte  0     : 0xDA header
      bytes 1-4   : class 0 GAP score, big-endian signed INT32
      ...
      bytes 25-28 : class 6 GAP score, big-endian signed INT32

    Returns (scores, warning) matching the _read_kws_scores convention.
    """
    log = dut._log
    header = await _uart_read_byte(dut, timeout_cycles=timeout_cycles)
    if header != SCORE_HEADER:
        return None, f"UART score packet: expected header 0x{SCORE_HEADER:02X}, got 0x{header:02X}"
    scores = []
    for idx, name in enumerate(class_names[:7]):
        raw = 0
        for _ in range(4):
            b = await _uart_read_byte(dut, timeout_cycles=timeout_cycles)
            raw = (raw << 8) | b
        if raw >= 0x80000000:
            raw -= 0x100000000
        scores.append((idx, name, raw))
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


def _sample_spect_sram_gl(dut, n_samples=16):
    """Read the first n_samples bytes from the 4 spectrogram SRAM macros in GL mode.

    The PNL flattens the spectrogram_sram hierarchy into escaped instance names
    directly under chip_top.  Bank A (spect_write_sel=0) is the write target
    after reset; bank B is the write target after the first spect_done toggle.
    lo = addresses 0-1023, hi = addresses 1024-1999.

    Returns (results_dict, summary_str).  On complete failure returns (None, msg).
    """
    chip_top = getattr(dut, "u_chip_top", None)
    if chip_top is None:
        return None, "u_chip_top not found"

    instances = {
        "spect_A_lo": r"\i_chip_core.kws_inst.inst_specram.gen_spect_banks[0].inst_spectrogram_sram ",
        "spect_A_hi": r"\i_chip_core.kws_inst.inst_specram.gen_spect_banks[1].inst_spectrogram_sram ",
        "spect_B_lo": r"\i_chip_core.kws_inst.inst_specram.gen_spect_banks[0].inst_spectrogram_sram2 ",
        "spect_B_hi": r"\i_chip_core.kws_inst.inst_specram.gen_spect_banks[1].inst_spectrogram_sram2 ",
    }

    results = {}
    for label, path in instances.items():
        try:
            sram = getattr(chip_top, path)
            vals = []
            for i in range(n_samples):
                try:
                    cell = sram.mem[i].value
                except Exception:
                    vals.append(None)          # cell not accessible: stop probing this bank
                    break
                try:
                    vals.append(int(cell))     # defined value
                except Exception:
                    vals.append('X')           # X/Z cell — do NOT break; keep scanning
            results[label] = vals
        except Exception as e:
            results[label] = f"unavailable ({type(e).__name__})"

    summaries = []
    for label, vals in results.items():
        if isinstance(vals, str):
            summaries.append(f"{label}={vals}")
            continue
        probed = [v for v in vals if v is not None]
        x_cnt = sum(1 for v in probed if v == 'X')
        if not probed:
            summaries.append(f"{label}=inaccessible")
        elif x_cnt:
            head = " ".join('XX' if v == 'X' else (f"{v:02x}" if v is not None else "??")
                            for v in vals[:8])
            summaries.append(f"{label}=X({x_cnt}/{len(probed)})[{head}]")
        elif all(v == 0 for v in probed):
            summaries.append(f"{label}=ALL_ZERO")
        else:
            hex_str = " ".join('XX' if v == 'X' else (f"{v:02x}" if v is not None else "??")
                               for v in vals[:8])
            summaries.append(f"{label}=[{hex_str}]")

    return results, "  ".join(summaries)


def _class_label(name, cls):
    if name is None and cls is None:
        return "unavailable"
    return f"{name if name is not None else '?'} ({cls if cls is not None else '?'})"


async def _drive_pdm_from_pads(dut, pdm_bits):
    for bit in pdm_bits:
        dut.input_PAD.value = 0b10 | (bit & 1)
        await RisingEdge(dut.clk_PAD)
    dut.input_PAD.value = 0


async def _check_class_pad_stability(dut, expected_cls, class_names, n_cycles=20):
    """Poll KWS class pads for n_cycles and warn if the value ever changes."""
    glitches = []
    for _ in range(n_cycles):
        await RisingEdge(dut.clk_PAD)
        _, cls = _read_kws_pads(dut)
        if cls != expected_cls:
            glitches.append(cls)
    if glitches:
        bad = sorted({class_names[c] if c < len(class_names) else f"cls{c}" for c in glitches})
        expected_name = class_names[expected_cls] if expected_cls < len(class_names) else f"cls{expected_cls}"
        dut._log.warning(
            f"  Class pad UNSTABLE over {n_cycles} cycles: glitched to {bad}"
            f" (expected stable {expected_name} ({expected_cls}))"
        )
    else:
        stable_name = class_names[expected_cls] if expected_cls < len(class_names) else f"cls{expected_cls}"
        dut._log.info(f"  Class pad stable over {n_cycles} cycles: {stable_name} ({expected_cls})")


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

@cocotb.test(skip=os.getenv("RUN_POWER", "0") != "1")
async def test_chip_top_power_window(dut):
    """
    Stripped-down workload optimized for SAIF generation on a small machine.
    
    Skips UART boot entirely — back-doors boot_done so the pipeline runs
    without flashing SRAMs. The chip won't produce meaningful KWS output,
    but switching activity through CIC / FIR / STFFT / LogMel / MAC array
    is representative for power analysis.
    
    Run with: RUN_POWER=1 GL=1 make sim
    """
    await start_up(dut)
    
    dut._log.info("=== Power-window test: skipping UART boot, forcing boot_done ===")
    
    # boot_done was flattened during synthesis. Its hierarchical name became
    # the escaped Verilog identifier "\i_chip_core.boot_done " (the trailing
    # space is part of the escape, not optional).
    try:
        boot_done_handle = getattr(dut.u_chip_top, r"\i_chip_core.boot_done ")
        boot_done_handle.value = 1
        dut._log.info("  boot_done forced high via flattened escape name")
    except (AttributeError, ValueError) as e:
        children = [n for n in dir(dut.u_chip_top) if not n.startswith("_")][:50]
        raise AssertionError(
            f"Could not force boot_done: {e}\n"
            f"  Sample children of u_chip_top: {children}"
        )
    
    await ClockCycles(dut.clk_PAD, 100)
    
    # Short synthetic chirp — ~200 PCM samples = ~12.5 ms of audio.
    # Long enough for a few STFFT frames in steady state.
    import numpy as np
    n = 200
    t = np.arange(n) / 16_000
    chirp = np.sin(2*np.pi*(500*t + 3000/2*t**2)) * 16000
    pcm = np.clip(chirp.astype(np.int32), -32768, 32767)
    pdm_bits = _pcm_to_pdm(pcm)
    
    dut._log.info(f"  Driving {len(pdm_bits)} PDM bits (~{len(pdm_bits)*62.5/1e6:.1f}ms sim)")
    
    # The wrapper IS the top in GL mode, so power_dump_en lives at dut.power_dump_en.
    # Begin VCD dump just before we start the audio stimulus.
    try:
        dut.power_dump_en.value = 1
        dut._log.info("  power_dump_en asserted — VCD dump started")
    except AttributeError:
        dut._log.warning("  power_dump_en not found at top — VCD dump may not trigger")
    
    await ClockCycles(dut.clk_PAD, 10)
    
    for bit in pdm_bits:
        dut.input_PAD.value = 0b10 | (bit & 1)
        await RisingEdge(dut.clk_PAD)
    dut.input_PAD.value = 0
    
    # Let the pipeline drain — a few mel-spectrogram frames should propagate
    # through during this window so we capture downstream KWS activity too.
    await ClockCycles(dut.clk_PAD, 50_000)
    
    try:
        dut.power_dump_en.value = 0
        await Timer(1, "ns")
    except AttributeError:
        pass
    
    dut._log.info("=== Power-window test complete ===")


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
    if gl:
        dut._log.info(f"  [gl_milestones] {_gl_milestone_str(dut)}")

    handles = _resolve_probe_handles(dut)
    force_pad_only = os.getenv("KWS_PAD_ONLY", "1" if gl else "0").lower() in ("1", "true", "yes")
    frontend_probes_available = _frontend_probes_available(handles) and not force_pad_only
    if not frontend_probes_available:
        reason = (
            "forced by KWS_PAD_ONLY/GL mode"
            if force_pad_only else
            "internal frontend probes are unavailable"
        )
        dut._log.warning(
            f"  Using pad-only KWS_DONE wait ({reason}). "
            "Internal milestones will be logged only when probe hierarchy is reliable."
        )
    pulse_counts = {name: 0 for name in ['cic_valid', 'fir_valid', 'fft_sync', 'fft_valid', 'power_valid', 'filterbank_done', 'log_done', 'mel_valid', 'spect_done', 'kws_start']}
    milestones = {name: False for name in ['cic_valid', 'fir_valid', 'fft_sync', 'fft_sync_aligned', 'fft_valid', 'power_valid', 'filterbank_done', 'log_done', 'mel_valid', 'spect_done', 'kws_start']}
    frontend_timeout_cycles = len(pdm_bits) + 200_000
    kws_timeout_cycles = len(pdm_bits) + 20_000_000
    monitor_cycles = frontend_timeout_cycles if frontend_probes_available else kws_timeout_cycles
    log_every = int(os.getenv("KWS_LOG_EVERY", "500000"))
    t0_real = time.time()
    frontend_cyc = None
    kws_started_cyc = None
    rtl_class = None
    kws_done_seen = False
    forensics = _GLForensics(dut, dut._log, len(pdm_bits)) if gl else None

    for cyc in range(monitor_cycles):
        if cyc < len(pdm_bits):
            dut.input_PAD.value = 0b10 | (pdm_bits[cyc] & 1)
        else:
            dut.input_PAD.value = 0
        await RisingEdge(dut.clk_PAD)

        if forensics is not None:
            forensics.tick(cyc + 1)

        if frontend_probes_available:
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
            if gl:
                _, spect_snap = _sample_spect_sram_gl(dut)
                dut._log.info(f"  [kws_done] spect SRAM snapshot: {spect_snap}")
            break
        if frontend_probes_available and kws_started_cyc is not None:
            frontend_cyc = cyc + 1
            break

        if cyc % log_every == log_every - 1:
            pdm_status = 'done' if cyc + 1 >= len(pdm_bits) else f'in flight ({cyc + 1}/{len(pdm_bits)})'
            elapsed_s = time.time() - t0_real
            elapsed_str = f"{elapsed_s / 60:.1f}min ({elapsed_s:.0f}s)"
            if frontend_probes_available:
                dut._log.info(f"  [chip_top heartbeat] cyc={cyc + 1:,} sim={(cyc + 1) * CLK_PERIOD_NS / 1e6:.1f}ms elapsed={elapsed_str} PDM={pdm_status} counts={pulse_counts} probes={_probe_str(dut)}")
            else:
                gl_state = f" | {_gl_milestone_str(dut)}" if gl else ""
                dut._log.info(f"  [chip_top heartbeat] cyc={cyc + 1:,} sim={(cyc + 1) * CLK_PERIOD_NS / 1e6:.1f}ms elapsed={elapsed_str} PDM={pdm_status} waiting for KWS_DONE pad{gl_state}")

    if frontend_cyc is None:
        if frontend_probes_available:
            missing = next((name for name, seen in milestones.items() if not seen), None)
            raise AssertionError(f"frontend never produced kws_start within {frontend_timeout_cycles:,} cycles; first missing milestone: {missing}; counts={pulse_counts}; probes={_probe_str(dut)}")
        if forensics is not None:
            forensics.summary(monitor_cycles)
        gl_state = f" Last GL milestones: {_gl_milestone_str(dut)}" if gl else ""
        first_x = ""
        if forensics is not None and forensics.firstx:
            lbl, c, prevv = forensics.firstx[0]
            first_x = f" FIRST corruption: {lbl} ({_fmt_val(prevv)}->X) at cyc={c:,}."
        raise AssertionError(
            f"KWS_DONE pad never asserted within {kws_timeout_cycles:,} cycles; "
            "pad-only completion checking was used, so this timeout is based "
            f"only on the chip_top output pads.{gl_state}{first_x}"
        )

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

    if gl:
        # Read gap scores directly from FSM accumulator registers preserved in the PNL.
        # The score TX sequencer and debug_gap*_test wires were added to RTL after tapeout
        # and are not present in the PNL netlist.
        pnl_scores = _gl_read_gap_scores_pnl(dut, dut._log)
        if pnl_scores is not None:
            dut._log.info(
                "  [gl_gap_scores] from global_pool_acc: "
                + "  ".join(f"cls{c}={v}" for c, v in pnl_scores)
            )
            scores = [(c, class_names[c] if c < len(class_names) else f"cls{c}", v)
                      for c, v in pnl_scores]
            score_warning = None
        else:
            dut._log.warning("  [gl_gap_scores] could not read global_pool_acc")
            scores, score_warning = _read_kws_scores(dut, class_names)

        _gl_probe_chip_core_signals(dut, dut._log)
        try:
            dut._log.info(
                f"  [gl_debug] bidir_PAD state: full={dut.bidir_PAD.value} "
                f"uart_tx_bit={_uart_tx_pad_bit(dut)}"
            )
        except Exception as e:
            dut._log.warning(f"  [gl_debug] bidir_PAD read failed: {e}")
    else:
        scores, score_warning = await _uart_read_score_packet(dut, class_names)
        if scores is None:
            dut._log.warning(f"  UART score packet failed ({score_warning}); falling back to hierarchy probe")
            scores, score_warning = _read_kws_scores(dut, class_names)

    await _check_class_pad_stability(dut, rtl_class, class_names)

    if gl:
        spect_data, spect_summary = _sample_spect_sram_gl(dut)
        if spect_data is None:
            dut._log.warning(f"  Spect SRAM probe: {spect_summary}")
        else:
            all_zero_banks = [
                k for k, v in spect_data.items()
                if isinstance(v, list) and all(x == 0 for x in v if x is not None)
            ]
            if all_zero_banks:
                dut._log.warning(
                    f"  Spect SRAM probe: ALL-ZERO banks {all_zero_banks} — {spect_summary}"
                )
                dut._log.warning(
                    "  ALL-ZERO spectrogram SRAM means KWS sees no input features; "
                    "classification is bias-only (likely root cause of wrong class)"
                )
            else:
                dut._log.info(f"  Spect SRAM probe: {spect_summary}")

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


@cocotb.test(skip=gl and os.getenv("RUN_GL_PAD_SMOKE", "0") != "1")
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

    if gl:
        milestones = _gl_milestone_str(dut)
        dut._log.info(f"[gl_milestones] {milestones}")
        assert "unavailable" not in milestones and "=?" not in milestones, \
            f"GL milestone probe nets missing from netlist: {milestones}"

    logger.info("chip_top pad smoke completed")


def link_readmemh_files(proj_path):
    rtl = (proj_path / "../src/rtl").resolve()
    links = [
        rtl / "STFFT/tests/hanning.hex",
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

    top_module = hdl_toplevel
    sources = []
    defines = {f"SLOT_{slot.upper()}": True, "SIM": True}
    includes = [proj_path / "../src/"]
    build_args = []                              # ← add this back

    if gl:
        sim_dir = proj_path / "sim"
        sources += [
            sim_dir / "gf180mcu_as_sc_mcu7t3v3.v",
            sim_dir / "gf180mcu_as_sc_mcu7t3v3_missing_cells.v",
            sim_dir / "gf180mcu_fd_io.v",
            sim_dir / "gf180mcu_ws_io.v",
            sim_dir / "gf180mcu_ocd_ip_sram_models.v",
            # SDF back-annotation requires the synthesis netlist (final/nl):
            # its instance names (e.g. _069513_) match the SDF, whereas the
            # PnR netlist (final/pnl) renumbers cells and leaves ~89% of SDF
            # INSTANCE paths unresolved ("Unable to find _NNNNNN_ in scope").
            # Plain GL (no SDF) keeps pnl.v, whose net names the probe helpers
            # (_gl_read_gap_scores_pnl, _gl_milestone_str) are verified against.
            proj_path / (f"../final/nl/{hdl_toplevel}.nl.v" if sdf_corner
                         else f"../final/pnl/{hdl_toplevel}.pnl.v"),
            sim_dir / "chip_top_sdf_wrapper.sv",
        ]
        top_module = "chip_top_sdf_wrapper"
        # FUNCTIONAL suppresses `specify` blocks in cell models via `ifndef FUNCTIONAL` guards.
        # For SDF back-annotation we need those blocks compiled in so $sdf_annotate has
        # paths to write into; omit FUNCTIONAL when a corner is requested.
        #
        # The SDF run targets the logical synthesis netlist (final/nl), which has no
        # power pins and no VDD/VSS top ports (verified: 0 power-pin refs vs 1.6M in
        # pnl). So USE_POWER_PINS must be OFF for it — cell models compile logic-only,
        # matching nl's pin-less instantiations — and the wrapper drops the .VDD/.VSS
        # binding under SDF_ENABLED to match nl's port list.
        if sdf_corner:
            defines = {}
        else:
            defines = {"USE_POWER_PINS": True, "FUNCTIONAL": True}
    else:
        sources.extend(rtl_sources(proj_path))

    if not gl:
        sources += [
            proj_path / "sim/gf180mcu_fd_io.v",
            proj_path / "sim/gf180mcu_ws_io.v",
        ]

    sources += [
        proj_path / "../ip/gf180mcu_ws_ip__qrcode_id/vh/gf180mcu_ws_ip__qrcode_id.v",
        proj_path / "../ip/gf180mcu_ws_ip__shuttle_id/vh/gf180mcu_ws_ip__shuttle_id.v",
        proj_path / "../ip/gf180mcu_ws_ip__project_id/vh/gf180mcu_ws_ip__project_id.v",
        proj_path / "../ip/gf180mcu_ws_ip__marker/vh/gf180mcu_ws_ip__marker.v",
        proj_path / "../ip/gf180mcu_ws_ip__logo/vh/gf180mcu_ws_ip__logo.v",
    ]

    # Icarus needs -g2012 to parse the .sv wrapper file in GLS mode
    if sim == "icarus" and gl:
        build_args += ["-g2012"]
    if sdf_corner:
        sdf_path = (proj_path / f"../final/sdf/{sdf_corner}/chip_top__{sdf_corner}.sdf").resolve()
        fixed_sdf_path = (proj_path / "sim_build" / f"chip_top__{sdf_corner}_fixed.sdf").resolve()
        sys.path.insert(0, str((proj_path / "../scripts").resolve()))
        import sdf_icarus_fixer
        (proj_path / "sim_build").mkdir(exist_ok=True)
        stats = sdf_icarus_fixer.fix(sdf_path, fixed_sdf_path)
        print(f"[sdf_fix] fixed {stats.triplets_fixed} triplet(s), "
              f"dropped {stats.interconnects_dropped} INTERCONNECT(s), "
              f"{stats.cells_dropped} CELL(s)  →  {fixed_sdf_path}")
        sdf_inc = proj_path / "sim" / "sdf_annotate.v"
        sdf_inc.write_text(f'initial $sdf_annotate("{fixed_sdf_path}", u_chip_top, , "sdf.log", "MAXIMUM");\n')
        build_args += ["-DSDF_ENABLED", f"-I{proj_path / 'sim'}", "-gspecify"]

    if sim == "verilator":
        build_args += ["--timing"]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel=top_module,
        defines=defines,
        always=True,
        includes=includes,
        build_args=build_args,
        waves=False,
    )
    link_readmemh_files(proj_path)

    plusargs = ["+notimingchecks"] if (gl and not sdf_corner) else []
    run_power_vcd = gl and os.getenv("RUN_POWER", "0").lower() in ("1", "true", "yes")
    if run_power_vcd:
        power_vcd_path = (proj_path / "sim_build" / "power_window.vcd").resolve()
        plusargs.append(f"+power_vcd_path={power_vcd_path}")

    # KWS_WAVES=1 arms a debug FST dump of u_chip_top (see chip_top_sdf_wrapper.sv).
    # KWS_DUMP_START_NS=<t> postpones the dump so long runs only capture the
    # window of interest instead of filling the disk.
    kws_waves = gl and os.getenv("KWS_WAVES", "0").lower() in ("1", "true", "yes")
    if kws_waves:
        fst_path = (proj_path / "sim_build" / "chip_top_gl.fst").resolve()
        plusargs.append(f"+kws_fst_path={fst_path}")
        dump_start_ns = os.getenv("KWS_DUMP_START_NS", "")
        if dump_start_ns:
            plusargs.append(f"+kws_dump_start_ns={dump_start_ns}")
        dump_stop_ns = os.getenv("KWS_DUMP_STOP_NS", "")
        if dump_stop_ns:
            plusargs.append(f"+kws_dump_stop_ns={dump_stop_ns}")

    if sim == "icarus" and (run_power_vcd or kws_waves):
        wave_flag = "-vcd" if run_power_vcd else "-fst"
        base_test_command = runner._test_command

        def _test_command_vcd():
            cmds = base_test_command()
            for cmd in cmds:
                for idx, arg in enumerate(cmd):
                    if arg == "-none":
                        cmd[idx] = wave_flag
            return cmds

        runner._test_command = _test_command_vcd

    runner.test(
        hdl_toplevel=top_module,
        test_module="chip_top_tb,",
        plusargs=plusargs,
        waves=False,
    )


if __name__ == "__main__":
    chip_top_runner()
