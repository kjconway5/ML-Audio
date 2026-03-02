# 256-point Short Time Fast Fourier Transform

  This directory contains the STFFT for the digital signal preprocessing pipeline. The STFFT is a form of the previously mentioned FFT, however in essence it computes multiple smaller FFT over certain windows. This allows the STFFT to determine audio signals with overlapping frequencies and frequencies that change over time, in our case speech. 

  Since we are now continuously moving a window across an incoming audio input to simulate many smaller FFTs across the signal. We must introduce different forms of measurement for the FFTs range. First we have the window size/frame size(in our case window and frame size are equal). The window size is the signal we are going to apply a singular FFT to. The window size can be measured in time and number of samples:

![Frame Size](src/rtl/STFFT/frame_size.png)


  Hop size, this is the time or amount of samples before a new window will begin. It is essential to have windows overlapping to prevent aliasing and errors in output(hop size is less than window size):

  Now it is important to keep in mind the time-frequency resolution of the STFFTs output. As the frame size increases the frequency resolution increase however the time resolution decreases. Additionally, the hop size dictates how many FFTs are computed in a given audio sound. Before we find suitable values for the window and hop size. I will denote that the FFT core will be 256-points to be accurate but efficient. 

  However to find the true frequency resolution we must look at the frame size. However, what do we denote as a good frame size or a bad one? Let us first look at our application: speech. It is found that the pitch period for speech is approximately 5-12ms(add citation). Since we want our output to have varying harmonics to gather data from we should choose double the max pitch period, 25ms. This works out as the range which speech appears stationary over smaller region is 20-30ms(add citation).

  Back to hop sizes, for a hop size we want to choose middle ground time period. If the hop size is to long(closer and closer to the frame size) there will be less FFTs computed resulting in a lower time resolution. However if we were to have an extremely small hop size this would be computationally expensive but also decrease the frequency resolution. Knowing this we must use speech dynamics to determine the range of a potential hop size. Earlier we mentioned that pitch periods range from 5-12ms. Since this period denotes areas of similar signals we can pick safely pick $\simeq 10ms$ as the hop size. The frame and hop size are subject to slight alterations in the future.

## STFFT Overview

- **Authors**: Michael Aguero: STFFT Wrapper RTL & Verfication, Jose Peralta Window Function Verfication
- **Architecture**: 256-point SFFT using radix-2 decimation-in-frequecy
- **Input/Output**: 16-bit complex samples (16-bit real and imaginary components)
- **Clock**: TBD
- **Processing Time**: TBD

## 256-point FFT Core

Gisselquist Technology's ZipCPU created an open source pipelined FFT generator, this allows us to generate a custom FFT core for the ASIC. This can be done by downloading and building the https://github.com/ZipCPU/dblclockfft that is available on GitHub by ZipCPU. Once the `make` command is complete there will be a `./fftgen` executable in the `sw/` directory. Using this executable alongside parameters we can build the custom core:

`./fftgen -f 256 -n 14 -m 18 -k 4 -p 1`

Here is a list of what the parameters do:

* -f: sets N, the sample size. In our case we have a 256 point FFT
* -n: sets the input width. In our case we have a 24 bit input width
* -x: increases the twiddle factor bit size. In our case we have a twiddle factor bit length of 14 (increasing the bit length for the Twiddle Factor helps prevent truncation errors)
* -p: sets the numbers of DSPs used. In our case we are using X DSPs

## Windowing Function

## Running Testbench

## Running Simulation

To run the RTL simulation:

```sh
make -B
```

To run gatelevel simulation, first harden your project and copy `../runs/wokwi/results/final/verilog/gl/{your_module_name}.v` to `gate_level_netlist.v`.

Then run:

```sh
make -B GATES=yes
```

If you wish to save the waveform in VCD format instead of FST format, edit tb.v to use `$dumpfile("tb.vcd");` and then run:

```sh
make -B FST=
```

This will generate `tb.vcd` instead of `tb.fst`.

## How to view the waveform file

Using GTKWave

```sh
gtkwave tb.fst tb.gtkw
```

Using Surfer

```sh
surfer tb.fst
```
