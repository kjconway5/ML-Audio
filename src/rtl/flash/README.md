# Flash / UART Boot Subsystem

The `flash` RTL implements the UART-based boot path for the ML audio chip. This
path is used to load runtime tables, model parameters, and control commands
before inference starts.

Despite the folder name, this block does not currently model an external
nonvolatile flash device. Instead, it receives framed UART packets from the host,
checks the packet contents, and routes valid payload bytes into on-chip SRAM
write ports used by the feature pipeline and DS-CNN.

At the full-chip level, `chip_core.sv` connects the physical UART RX/TX blocks
to `boot_controller`. Valid boot writes are then routed through
`features_boot_router` and `dscnn_boot_router`. The feature pipeline stays in
boot reset until the host sends the `CTRL_BOOT_DONE` control packet.

## Folder Contents

| File | Purpose |
|---|---|
| `boot_pkg.sv` | Shared boot constants, target encodings, packet byte values, and the `boot_bus_t` definition. |
| `boot_controller.sv` | UART byte-stream packet parser. Checks sync bytes and checksum, emits ACK/NACK responses, and converts payload bytes into boot-bus writes. |
| `features_boot_router.sv` | Routes feature-pipeline boot writes to the Log LUT, Mel coefficients, Mel metadata, and VAD threshold register. |
| `dscnn_boot_router.sv` | Routes DS-CNN boot writes to weight SRAM, bias SRAM, and layer configuration storage. |
| `flash_top.sv` | Simulation-only harness with behavioral memories and read-back ports for verifying the controller and boot routers. |
| `test_flash_top.py` | Cocotb test that writes known data into the simulated memories and reads it back for verification. |
| `simple_flash.sv` | Older standalone UART byte-capture SRAM helper. This is separate from the packetized boot path. |
| `uart.v`, `uart_rx.v`, `uart_tx.v` | UART IP used for chip-level integration and standalone UART-based helpers. |
| `Makefile` | Cocotb/Icarus entry points for testing `flash_top` and `boot_controller`. |

## Packet Format

Each host packet uses the following frame format:

```text
0xAA 0x55
target
addr_hi addr_lo
len_hi  len_lo
payload[0:len-1]
checksum
```

The checksum is an XOR over the following fields:

```text
target
addr_hi
addr_lo
len_hi
len_lo
payload bytes
```

The two sync bytes, `0xAA` and `0x55`, are not included in the checksum.

After the checksum byte is received, the controller responds with one byte:

| Response | Value | Meaning |
|---|---:|---|
| ACK | `0x06` | Packet accepted. |
| NACK | `0xEE` | Checksum failed. |
| ERR | `0xE1` | Reserved for malformed-packet handling. |

`boot_controller` currently accepts input bytes whenever `rx_valid_i` is high,
since `rx_ready_o` is tied high.

## Target Encoding

The `target` byte is split into a module nibble and a subtarget nibble:

```text
target[7:4] = module
target[3:0] = subtarget
```

### Module Map

| Module | Nibble | Description |
|---|---:|---|
| `MOD_FEATURES` | `0x0` | Feature-pipeline tables and thresholds. |
| `MOD_DSCNN` | `0x1` | DS-CNN weights, biases, and layer configuration. |
| `MOD_DEBUG` | `0x8` | Reserved/debug address space. |
| `MOD_CONTROL` | `0xF` | Boot and runtime control commands. |

### Feature Subtargets

| Subtarget | Nibble | Width | Destination |
|---|---:|---:|---|
| `FEAT_LOG_LUT` | `0x0` | 16-bit | Log lookup table, 64 words. |
| `FEAT_MEL_COEFF` | `0x1` | 16-bit | Sparse Mel coefficient SRAM, 256 words. |
| `FEAT_MEL_META` | `0x2` | 8-bit | Mel start/end/offset metadata. |
| `FEAT_VAD_THRESH` | `0x3` | 16-bit | VAD threshold register. |

Hann window values and FFT twiddles are still hard ROMs loaded through
`$readmemh` in the STFFT/FFT RTL. They are not currently loadable through this
UART boot path.

### DS-CNN Subtargets

| Subtarget | Nibble | Width | Destination |
|---|---:|---:|---|
| `DSCNN_WEIGHTS` | `0x0` | 8-bit | INT8 weight SRAM. |
| `DSCNN_CFG` | `0x1` | 8-bit | Layer configuration registers/SRAM. |
| `DSCNN_BIAS` | `0x2` | 8-bit | Byte-addressed INT32 bias SRAM. |

### Control Subtargets

| Subtarget | Nibble | Effect |
|---|---:|---|
| `CTRL_BOOT_DONE` | `0x0` | Asserts `boot_done_o` after ACK, releasing inference reset in `chip_core.sv`. |
| `CTRL_START` | `0x1` | Reserved for a future explicit inference-start command. |

## Payload Packing

The controller uses `is_target_16bit()` from `boot_pkg.sv` to decide how payload
bytes should be converted into boot-bus writes.

For 8-bit targets, each payload byte becomes one write:

```text
payload byte -> boot_i.data[7:0]
write_addr increments by 1 per byte
```

For 16-bit targets, the host sends little-endian byte pairs:

```text
payload[0] = word[7:0]
payload[1] = word[15:8]
```

The controller writes the packed value as:

```text
{payload[1], payload[0]}
```

and increments the write address once per 16-bit word.

## Boot Sequence

A normal chip boot uses the following sequence:

1. Load feature-pipeline data:
   - Log LUT
   - Mel coefficients
   - Mel metadata

2. Load DS-CNN data:
   - weights
   - biases
   - layer configuration

3. Optionally write feature/control registers, such as the VAD threshold.

4. Send `MOD_CONTROL:CTRL_BOOT_DONE` with a zero-length payload.

After `CTRL_BOOT_DONE` is ACKed, `boot_done_o` remains high. In `chip_core.sv`,
this releases the feature pipeline from boot reset. The KWS path remains in its
normal reset/start flow and cannot begin inference until the post-boot feature
pipeline produces `spect_done`.

## Simulation

Run the flash subsystem cocotb test from this directory:

```sh
cd src/rtl/flash
make test-cocotb
```

This builds `flash_top.sv`, sends valid packets for each supported memory type,
checks for ACK responses, asserts boot done, and reads the behavioral memories
back through the verification ports.

To run the standalone `boot_controller` target through the same Makefile:

```sh
cd src/rtl/flash
make test-boot-controller
```

Use the following command to remove simulation products:

```sh
make clean
```

## Integration Notes

- `flash_top.sv` is only a simulation harness. Real chip integration happens in
  `chip_core.sv`.
- The packet controller operates on a byte-stream interface. Physical UART
  serialization and deserialization are handled outside the controller.
- Unknown module or subtarget values are ignored by the routers after the packet
  is ACKed. They do not currently generate `ERR`.
- `dscnn_boot_router.sv` supports bias writes, but the local `flash_top` test
  harness only exposes read-back checks for weights and layer configuration.