import cocotb
from cocotb.clock import Clock, Timer
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles

# Expected ROM contents from the Verilog case statement
EXPECTED_HEX = {
    0:  0x0005DAB4,
    1:  0xFFFB6A5B,
    2:  0x0003D0CB,
    3:  0xFFFFF589,
    4:  0x0004EEF5,
    5:  0x0005A169,
    6:  0x000585D1,
    7:  0xFFFB1AF9,
    8:  0x000555B6,
    9:  0xFFFA892C,
    10: 0x0002E223,
    11: 0x000374CA,
    12: 0xFFFBFC59,
    13: 0x000285CC,
    14: 0xFFFC43A9,
    15: 0xFFFC35E4,
    16: 0x00036189,
    17: 0x00039109,
    18: 0x0003A74F,
    19: 0x0001F4B3,
    20: 0x0005E39D,
    21: 0x0003BFE0,
    22: 0x00004B93,
    23: 0x0009A8FD,

    24: 0xFFFFFDD1,
    25: 0xFFFFFD37,
    26: 0x0000028A,
    27: 0x0000054F,
    28: 0x00000405,
    29: 0xFFFFFDB8,
    30: 0x0000077E,
    31: 0x0000046B,
    32: 0x00000775,
    33: 0x0000030B,
    34: 0xFFFFFE6D,
    35: 0x0000152A,
    36: 0x00000047,
    37: 0x000002CE,
    38: 0xFFFFFD08,
    39: 0xFFFFFE0A,
    40: 0x0000093B,
    41: 0x00000357,
    42: 0xFFFFFD81,
    43: 0x000001E8,
    44: 0x000003E5,
    45: 0x00000A51,
    46: 0x000003E1,
    47: 0x00000386,

    48: 0xFFFFFD7B,
    49: 0xFFFFDCBB,
    50: 0x0000005D,
    51: 0x00000411,
    52: 0x000010F9,
    53: 0xFFFFE9E5,
    54: 0xFFFFED73,
    55: 0x00002A76,
    56: 0xFFFFE866,
    57: 0x00001BF1,
    58: 0x0000241B,
    59: 0x00001402,
    60: 0x0000003C,
    61: 0x000001BD,
    62: 0x00000903,
    63: 0x00002096,
    64: 0x0000043B,
    65: 0x00001981,
    66: 0x00002FE3,
    67: 0x00002091,
    68: 0x00003437,
    69: 0x00001012,
    70: 0x000011C0,
    71: 0xFFFFF55C,

    72: 0xFFFFFD1A,
    73: 0x00000572,
    74: 0xFFFFFEDD,
    75: 0x000004EA,
    76: 0xFFFFFD1F,
    77: 0x0000064A,
    78: 0xFFFFFB7A,
    79: 0xFFFFFE55,
    80: 0x00000635,
    81: 0xFFFFF938,
    82: 0x0000062D,
    83: 0xFFFFFF8C,
    84: 0x0000055D,
    85: 0xFFFFFF77,
    86: 0x00000614,
    87: 0xFFFFFEB6,
    88: 0x00000085,
    89: 0x00000452,
    90: 0x00000127,
    91: 0xFFFFFFEE,
    92: 0x000008CF,
    93: 0x00000460,
    94: 0x0000046B,
    95: 0x00000004,

    96:  0xFFFFFCCA,
    97:  0x00001B43,
    98:  0x0000034F,
    99:  0x00001192,
    100: 0x000003B3,
    101: 0xFFFFFFD8,
    102: 0x00000134,
    103: 0x00000FF7,
    104: 0xFFFFE8B0,
    105: 0x000005CE,
    106: 0x000003FC,
    107: 0x000009ED,
    108: 0x00000ED8,
    109: 0x000008C2,
    110: 0xFFFFEDAB,
    111: 0x00000387,
    112: 0x000019A0,
    113: 0xFFFFE785,
    114: 0x00000588,
    115: 0xFFFFFF57,
    116: 0x0000029A,
    117: 0x00000F20,
    118: 0x000007E7,
    119: 0x000000F0,

    120: 0xFFFFFDA1,
    121: 0xFFFFFC65,
    122: 0xFFFFFF8E,
    123: 0x00000299,
    124: 0x0000023A,
    125: 0x000003A0,
    126: 0xFFFFFEF4,
    127: 0x0000064B,
    128: 0xFFFFFDE5,
    129: 0x000004CA,
    130: 0xFFFFFDAC,
    131: 0xFFFFFDF7,
    132: 0xFFFFFAEF,
    133: 0xFFFFFC49,
    134: 0x000006A3,
    135: 0xFFFFFEF4,
    136: 0xFFFFFF5E,
    137: 0x000003F4,
    138: 0xFFFFFF8D,
    139: 0xFFFFFE64,
    140: 0x00000757,
    141: 0x0000029F,
    142: 0xFFFFFD12,
    143: 0xFFFFFE88,

    144: 0x0000013D,
    145: 0xFFFFFEA5,
    146: 0xFFFFF964,
    147: 0x00000158,
    148: 0xFFFFF744,
    149: 0xFFFFFC73,
    150: 0x00000393,
    151: 0x00000177,
    152: 0xFFFFFCBC,
    153: 0x00000147,
    154: 0x00000AF7,
    155: 0x000003BE,
    156: 0x0000009D,
    157: 0xFFFFF312,
    158: 0x00000173,
    159: 0x00000202,
    160: 0x0000007C,
    161: 0xFFFFF585,
    162: 0x000005E4,
    163: 0xFFFFFA67,
    164: 0x0000018A,
    165: 0xFFFFFA56,
    166: 0xFFFFFBF6,
    167: 0xFFFFF9F6,

    168: 0xFFFFFC59,
    169: 0x0000030F,
    170: 0xFFFFFC8B,
    171: 0xFFFFFCB7,
    172: 0x0000035A,
    173: 0xFFFFFE55,
    174: 0xFFFFFF15,
    175: 0xFFFFFFBD,
    176: 0xFFFFFAFF,
    177: 0xFFFFFCBA,
    178: 0x00000149,
    179: 0x00000264,
    180: 0xFFFFFEC9,
    181: 0xFFFFFD19,
    182: 0xFFFFF9C9,
    183: 0xFFFFFE16,
    184: 0xFFFFFD57,
    185: 0x000001A8,
    186: 0x000002C7,
    187: 0xFFFFFBF0,
    188: 0x000002DA,
    189: 0xFFFFFE1F,
    190: 0xFFFFFF01,
    191: 0xFFFFFBB1,

    192: 0xFFFFFFB8,
    193: 0x00000004,
    194: 0xFFFFFF7E,
    195: 0xFFFFFFCE,
    196: 0xFFFFFF29,
    197: 0xFFFFFFB7,
    198: 0x000000CB,
    199: 0x0000016D,
    200: 0xFFFFFF88,
    201: 0xFFFFFF48,
    202: 0x00000009,
    203: 0xFFFFFFC9,
    204: 0x00000210,
    205: 0x0000004A,
    206: 0xFFFFFFD5,
    207: 0x000000B1,
    208: 0xFFFFFF62,
    209: 0xFFFFFFDA,
    210: 0xFFFFFF62,
    211: 0xFFFFFFB5,
    212: 0xFFFFFF90,
    213: 0x00000047,
    214: 0x00000092,
    215: 0xFFFFFF78,

    216: 0xFFFFFFFF,
    217: 0x00000012,
    218: 0xFFFFFFC0,
    219: 0xFFFFFFA8,
    220: 0x00000085,
    221: 0x00000001,
    222: 0xFFFFFFF7,
}

DEPTH = 223
DATA_W = 32

async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    Also resets energies to 0 and mel_idx_i to 0.
    """
    dut.reset.value = 1
    dut.log_en_i.value = 0
    dut.mel_idx_i.value = 0
    # reset energies
    for i in range(len(dut.mel_energy_i)):
        dut.mel_energy_i[i].value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)


@cocotb.test()
async def test_bias_dffs_basic_addresses(dut):
    test_addrs = [
        0,    # first_conv first
        1,    # negative
        23,   # first_conv last
        24,   # ds_blocks.0.depthwise first
        48,   # ds_blocks.0.pointwise first
        71,   # ds_blocks.0.pointwise last
        72,   # ds_blocks.1.depthwise first
        96,   # ds_blocks.1.pointwise first
        120,  # ds_blocks.2.depthwise first
        144,  # ds_blocks.2.pointwise first
        168,  # ds_blocks.3.depthwise first
        192,  # ds_blocks.3.pointwise first
        216,  # classifier first
        222,  # classifier last valid
        223,  # default case
        255,  # default case
    ]

    for addr in test_addrs:
        dut.addr.value = addr
        await Timer(1, units="ns")

        got = dut.data.value.signed_integer
        
        raw = EXPECTED_HEX.get(addr, 0)
        
        if raw & (1 << 31):
            exp = raw - (1 << 32)
        else:
            exp = raw

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_dffs_basic_addresses passed")


@cocotb.test()
async def test_bias_dffs_full_sweep(dut):
    for addr in range(256):
        dut.addr.value = addr
        await Timer(1, units="ns")

        got = dut.data.value.signed_integer
        
        raw = EXPECTED_HEX.get(addr, 0)
        
        if raw & (1 << 31):
            exp = raw - (1 << 32)
        else:
            exp = raw

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_dffs_full_sweep passed")


@cocotb.test()
async def test_bias_dffs_block_boundaries(dut):
    boundary_addrs = [
        0, 23,      # first_conv
        24, 47,     # ds_blocks.0.depthwise
        48, 71,     # ds_blocks.0.pointwise
        72, 95,     # ds_blocks.1.depthwise
        96, 119,    # ds_blocks.1.pointwise
        120, 143,   # ds_blocks.2.depthwise
        144, 167,   # ds_blocks.2.pointwise
        168, 191,   # ds_blocks.3.depthwise
        192, 215,   # ds_blocks.3.pointwise
        216, 222,   # classifier
    ]

    for addr in boundary_addrs:
        dut.addr.value = addr
        await Timer(1, units="ns")

        got = dut.data.value.signed_integer
        
        raw = EXPECTED_HEX.get(addr, 0)
        
        if raw & (1 << 31):
            exp = raw - (1 << 32)
        else:
            exp = raw

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_dffs_block_boundaries passed")