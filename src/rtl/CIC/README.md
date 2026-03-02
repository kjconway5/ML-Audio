
Directory Structure Summary

rtl/<module_name>/
├── <module_name>.sv      # RTL design
├── tb_<module_name>.sv   # SystemVerilog testbench
├── test_<module_name>.py # Cocotb Python tests
├── test_runner.py        # Pytest runner ()
├── filelist.json         # File manifest
├── Makefile              # Build configuration
└── README.md             # (optional) Module documentation


Since the PDM microphone will be outputting
a high-frequency 1-bit stream and our pro-
posed design will require a 16-bit output at
a sampling rate of 16kHz. We must convert
the pulse-density modulation into pulse-code
modulation(the standard form of digital audio
in hardware). This can be done by reducing the
sampling rate(decimation) and increasing the
word length, the preferred method to do this
is a Cascaded Integrator Comb Filter.
The filter consists of an equal number of in-
tegrator and comb filters alongside a decimator.
This describes a single stage CIC filter, however
we can improve attenuation by increasing the
stages of integrators and comb filters.
For this design we choose the number of
stages to be N = 5, this provides a good spot.
We can analyze this decision by looking at the
magnitude response and transfer function of
the CIC Filter:
H(z) = [(1-z^(-R)/(1-z^(-1))]^N

In these equations M = 1 and R = fi/fo. Additionally,
upon analyzing these equations in MATLAB we
find that the outputted frequencies will have
less gain as the frequencies increases. This in
turn will cause the higher frequencies bins on
a spectrogram(input to ML model) to be low
and potentially cause issues for ML speech
detection, this is called pass-band droop. To
combat this we can employ compensation FIR
Filters since ”in typical decimation filtering ap-
plication we desire reasonably flat pass band
gain and narrow transition region width per-
formance”.
