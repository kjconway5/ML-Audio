"""Cocotb testbench skeleton for flash_top.sv

Simulates a full flash, reads each ram 
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

#Protocal Constants
ACK_BYTE  = 0x06
NACK_BYTE = 0xEE

# Module select (high nibble of target byte)
MOD_FEATURES = 0x0
MOD_DSCNN    = 0x1
MOD_CONTROL  = 0xF

# Features subtargets
FEAT_LOG_LUT   = 0x0  # 16-bit
FEAT_MEL_COEFF = 0x1  # 16-bit
FEAT_MEL_META  = 0x2  # 8-bit
FEAT_HANN      = 0x3  # 16-bit
FEAT_TWIDDLES  = 0x4  # 16-bit

# DS-CNN subtargets
DSCNN_WEIGHTS  = 0x0  # 8-bit
DSCNN_CFG      = 0x1  # 8-bit

# Control subtargets
CTRL_BOOT_DONE = 0x0
def make_target(mod, sub):
    return ((mod & 0xF) << 4) | (sub & 0xF)


#Helper Functions

async def reset_dut(dut):
    """Apply synchronous reset for a few cycles."""
    dut.reset.value      = 1
    dut.rx_byte_i.value  = 0
    dut.rx_valid_i.value = 0
    dut.tx_ready_i.value = 1
    dut.verify_lut_addr_i.value    = 0
    dut.verify_mel_addr_i.value    = 0
    dut.verify_meta_addr_i.value   = 0
    dut.verify_weight_addr_i.value = 0
    dut.verify_cfg_addr_i.value    = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def send_byte(dut, byte):
    #Push one byte via the UART byte interface (.
    await RisingEdge(dut.clk)
    dut.rx_byte_i.value  = byte & 0xFF
    dut.rx_valid_i.value = 1
    await RisingEdge(dut.clk)
    dut.rx_valid_i.value = 0
    dut.rx_byte_i.value  = 0


async def send_packet(dut, target, addr, payload):
    #Frame and send: AA 55 | target | addr_hi addr_lo | len_hi len_lo | payload | cksum.
    length = len(payload)
    #Combines bytes, and splits addr + leng to their high byte and low bytes
    body = [target, (addr >> 8) & 0xFF, addr & 0xFF,
            (length >> 8) & 0xFF, length & 0xFF] + list(payload)
    cksum = 0
    #Check Sum
    for b in body:
        cksum ^= b
    #Sync Bytes    
    await send_byte(dut, 0xAA)
    await send_byte(dut, 0x55)
    #Send rest of bytes
    for b in body:
        await send_byte(dut, b)
    await send_byte(dut, cksum & 0xFF)


async def wait_for_tx(dut, timeout=200):
    """Wait for tx_valid_o pulse; returns the byte (ACK/NACK)."""
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if int(dut.tx_valid_o.value) == 1:
            return int(dut.tx_byte_o.value)
    raise TimeoutError("tx_valid_o never asserted")


async def verify_read(dut, addr_sig, data_sig, addr):
    """Drive verify address, wait for 1-cycle read latency, sample data."""
    addr_sig.value = addr
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    return int(data_sig.value)

async def full_flash(dut, lut_data, mel_data, meta_data, weight_data, cfg_data):
    #flash all SRAMs in order, verify ACK after each packet, send boot_done at the end.


    async def flash_16bit(target, addr, words):
        #Pack 16-bit words into lo/hi byte pairs and send as one burst.
        payload = []
        for w in words:
            payload.append(w & 0xFF)         # lo byte first
            payload.append((w >> 8) & 0xFF)  # hi byte second
        await send_packet(dut, target, addr, payload)
        resp = await wait_for_tx(dut)
        assert resp == ACK_BYTE, f"target 0x{target:02X}: expected ACK, got 0x{resp:02X}"

    async def flash_8bit(target, addr, data):
        #Send raw bytes as one burst.
        await send_packet(dut, target, addr, data)
        resp = await wait_for_tx(dut)
        assert resp == ACK_BYTE, f"target 0x{target:02X}: expected ACK, got 0x{resp:02X}"

    #Flash each target in sequence
    await flash_16bit(make_target(MOD_FEATURES, FEAT_LOG_LUT),   0, lut_data)
    await flash_16bit(make_target(MOD_FEATURES, FEAT_MEL_COEFF), 0, mel_data)
    await flash_8bit (make_target(MOD_FEATURES, FEAT_MEL_META),  0, meta_data)
    await flash_8bit (make_target(MOD_DSCNN,    DSCNN_WEIGHTS),  0, weight_data)
    await flash_8bit (make_target(MOD_DSCNN,    DSCNN_CFG),      0, cfg_data)

    # Signal boot complete 
    await send_packet(dut, make_target(MOD_CONTROL, CTRL_BOOT_DONE), 0, [])
    resp = await wait_for_tx(dut)
    assert resp == ACK_BYTE, f"boot_done: expected ACK, got 0x{resp:02X}"
    await RisingEdge(dut.clk)
    assert int(dut.boot_done_o.value) == 1, "boot_done_o not asserted"

async def verify_full_flash(dut, lut_data, mel_data, meta_data, weight_data, cfg_data):
    #Read back every memory location and compare to expected data.

    for i, expected in enumerate(lut_data):
        got = await verify_read(dut, dut.verify_lut_addr_i, dut.verify_lut_data_o, i)
        assert got == expected, f"lut[{i}]: expected 0x{expected:04X}, got 0x{got:04X}"

    for i, expected in enumerate(mel_data):
        got = await verify_read(dut, dut.verify_mel_addr_i, dut.verify_mel_data_o, i)
        assert got == expected, f"mel[{i}]: expected 0x{expected:04X}, got 0x{got:04X}"

    for i, expected in enumerate(meta_data):
        got = await verify_read(dut, dut.verify_meta_addr_i, dut.verify_meta_data_o, i)
        assert got == expected, f"meta[{i}]: expected 0x{expected:02X}, got 0x{got:02X}"

    for i, expected in enumerate(weight_data):
        got = await verify_read(dut, dut.verify_weight_addr_i, dut.verify_weight_data_o, i)
        assert got == expected, f"weight[{i}]: expected 0x{expected:02X}, got 0x{got:02X}"

    for i, expected in enumerate(cfg_data):
        got = await verify_read(dut, dut.verify_cfg_addr_i, dut.verify_cfg_data_o, i)
        assert got == expected, f"cfg[{i}]: expected 0x{expected:02X}, got 0x{got:02X}"

#Tests


@cocotb.test()
async def test_full_flash(dut):
    #flash all SRAMs with known data, verify every location.
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Generate known test data (small subsets for speed)
    lut_data    = [(i * 0x0111) & 0xFFFF for i in range(64)]
    mel_data    = [(i * 0x0303) & 0xFFFF for i in range(640)]
    meta_data   = [(i * 3) & 0xFF for i in range(80)]
    weight_data = [(i * 7) & 0xFF for i in range(256)]  # subset, not full 4296
    cfg_data    = [(i * 11) & 0xFF for i in range(32)]

    await full_flash(dut, lut_data, mel_data, meta_data, weight_data, cfg_data)
    await verify_full_flash(dut, lut_data, mel_data, meta_data, weight_data, cfg_data)

    cocotb.log.info("test_full_flash PASSED")