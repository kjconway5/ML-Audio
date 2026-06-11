
# 256-point Short Time Fast Fourier Transform
The Short Time Fast Fourier Transform is a version of the Discrete Fourier Transform:
&emsp; &emsp; $F(n) = \sum_{k=0}^{N-1}f[k]e^\text{-j*(2pi/N)*nk}$
where instead of computing the computationally expensive DFT, $O(n^2)$, it utilizes the efficient radix-2 Fast Fourier Transform algorithm:  $O(nlogn)$. However, in the case of a human speech, the signal produced is aperiodic and the DFT treats a finite sequence as one period of an implicitly periodic signal[^2].

To solve the issue of transforming an aperiodic signal, the Short Time Fast Fourier Transform must be used. The STFFT is comprised of multiple smaller FFTs that span over a specified sample range instead of a single FFT. This is achieved through a layered windowing system[^3]: $h[n]=h_d[n]w[n]$

Now that there is a continuously moving window across an incoming audio input to simulate many smaller FFTs across the signal. We must introduce different forms of measurement for the FFTs range. First we have the window size/frame size(in our case window and frame size are equal). The window size is the sample range of the window function we are going to apply a singular FFT to:
  
<img src="ZipCPU/util/frame_size.png" alt="Sized Image" width="500" height="500"> [^1]

Hop size, this is the time or amount of samples before a new window will begin. It is essential to have windows overlapping to prevent aliasing and errors in output(hop size is less than window size):

  <img src="ZipCPU/util/hop_size.png" alt="Sized Image" width="500" height="500"> [^1]

It is important to keep in mind the time-frequency resolution of the STFFTs output. As the frame size increases the frequency resolution increase however the time resolution decreases. Additionally, the hop size dictates how many FFTs are computed in a given audio sound.

## STFFT Overview

- **Authors**: Michael Aguero
- **Architecture**: 256-point SFFT using radix-2 decimation-in-frequecy
- **Input/Output**: 16-bit complex samples (16-bit real and imaginary components)
- **Clock**: 16MHz
- **Processing Time**: ~11k Cycles
- **FFT Size**: 256
- **Window Size**: 256
- **Hop Size**: 128



## FFT Core IP: R2FFT by yoonisi [^4]

The R2FFT IP assumes each working-memory bank is a true dual-port (1R1W) SRAM and uses two banks in parallel to read an even- and an odd-indexed word every cycle. This delivers one butterfly per cycle but requires dual-port macros — roughly twice the silicon area of single-port macros, and not available in GF180MCU

Our redesign replaces this with two single-port banks held in ping-pong configuration. The two banks no longer split one frame into even/odd halves — they each hold a *complete, separate* frame. While one bank is being computed on, the other is being filled with the next frame or drained by DMA. The result is the same effective throughput at the STFFT-frame level, with roughly half the SRAM area and a single FFT core instead of two. The cost is per-frame compute latency: one butterfly now takes six cycles instead of one.

### Memory organization

| |  R2FFT Dual Port RAM | R2FFT Single Port RAM |
|---|---|---|
| Bank count | 2 | 2 |
| Macro type | Dual-port (1R1W) | Single-port |
| Bank contents | Even / odd halves of one frame | Full frame each |
| Bank role | Parallel (banking) | Alternating (ping-pong) |
| Butterfly cycles | ~1 | 6 |
| FFT cores per STFT | 2 (one per overlap channel) | 1 |

Each bank is 256 × 32 bits (16-bit real + 16-bit imaginary). Total FFT working memory: 8 single-port SRAM macros, with double-buffering subsumed into the ping-pong scheme.

### Per-RAM lifecycle

Each RAM runs an independent six-state machine:

```
IDLE → FILLING → READY_COMPUTE → COMPUTING → READY_DMA → DMAING → IDLE
```

| State | Activity |
|---|---|
| `IDLE` | RAM is empty and available to accept a new frame |
| `FILLING` | Input samples being bit-reverse-written into the bank |
| `READY_COMPUTE` | Frame complete, waiting for the shared butterfly engine |
| `COMPUTING` | Butterfly controller is reading and writing this bank |
| `READY_DMA` | FFT compute done, waiting for DMA read-out to start |
| `DMAING` | Bins streamed out to the downstream STFFT / LogMel chain |

A top-level sub-sequencer activates whenever any bank is in `COMPUTING` and walks the standard radix-2 stage loop (`SETUP → RUN → WAIT_PIPELINE → NEXT_STAGE → DONE`). Priority logic in the per-RAM transitions resolves the case where both banks want to start the same phase on the same cycle.

### Memory controller

The memory controller is a per-RAM combinational mux that selects the active port driver based on that bank's current state:

| Bank state | Port driven by |
|---|---|
| `FILLING` | Bit-reversed input writer |
| `COMPUTING` | Butterfly controller |
| `DMAING` | External DMA address |

Because each RAM has only one port, three concurrent operations (fill, compute, DMA) can coexist only because there are two banks: one services the butterfly while the other handles input or DMA in any given cycle.

### Butterfly controller

A single-port RAM cannot deliver both butterfly operands in one cycle. The butterfly controller serializes one radix-2 butterfly across six cycles:

```
S_READ_A → S_READ_B → S_LAUNCH → S_WAIT_COMPUTE → S_WRITE_A → S_WRITE_B
```

| Cycle | Action |
|---|---|
| `S_READ_A` | Issue read of address A; result returns next cycle |
| `S_READ_B` | Issue read of address B; first operand latched in `xa_lat` |
| `S_LAUNCH` | Both operands valid; fire `butterflyCore` with current twiddle |
| `S_WAIT_COMPUTE` | Drain `PL_DEPTH` pipeline registers in the radix-2 core |
| `S_WRITE_A` | Write the first result back to address A |
| `S_WRITE_B` | Write the second result back to address B |

The two butterfly addresses are computed inline from a butterfly counter `bf_cnt` and the current FFT stage index: A and B differ only at bit position `stageCount` (0 for A, 1 for B). The twiddle ROM address is `bit_reverse(bf_cnt >> stageCount)`. No external address generator is needed — this collapses the reference design's `fftAddressGenerator` into a few lines of combinational logic.

### Per-frame BFP exponent

Because frames overlap in the FFT (one filling while another computes while a third drains), each frame's block-floating-point exponent must survive until *its* DMA finishes. The controller maintains per-bank state:

```systemverilog
reg [FFT_BFPDW-1:0] ram0_init_bw, ram1_init_bw;   // latched at end of fill
reg signed [7:0]    ram0_bfpexp,  ram1_bfpexp;    // latched at end of compute
assign bfpexp = dma_target ? ram1_bfpexp : ram0_bfpexp;
```

The exported `bfpexp` is muxed by `dma_target` — whichever bank is currently in `DMAING` — so the downstream `bfpexp_for_mel` latch in `pipeline_top` always captures the exponent of the frame being emitted, even when three frames are in flight.

### Trade-offs

Single-port SRAM and frame-level pipelining cut working-memory area roughly in half compared to the dual-port reference, and remove the need for a second FFT core in the STFFT, at the cost of ~6× more cycles per butterfly (~11k cycles per 256-point frame versus ~5.5k for the reference). The architecture is sized so that frame compute fits inside the trigger interval `HOP × CE_EVERY`; the resulting timing constraint and the streaming handshake (`s_ready`) on the input port are documented in the STFFT pipeline section.


## Running Testbench

To run the FFT testbench:
```sh
cd R2FFT/test
make test-fft
```

To run the STFFT testbench:
```sh
cd tests
make test-stfft
```
[^1]: Velardo, V. (2021). "Short-Time Fourier Transform Explained Easily" *https://www.youtube.com/watch?v=-Yxj3yfvY-4*.

[^2]: Oppenhiem, A. & Schafer. R (2010). "Discrete-Time Signal Processing" *Chapter 7: Filter Design Techniques*.

[^3]: Ye, H. (2026). "UCSC ECE 153/250: Digital Signal Processing" *Lecture 14: FIR Filter Design*.

[^4]: yoonish (2021). "R2FFT: R2FFT is a fully synthesizable verilog module for doing the FFT on an FPGA or ASIC. " *https://github.com/yoonisi/R2FFT*.

