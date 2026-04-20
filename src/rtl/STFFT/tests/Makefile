TOP     := stfft
STFFT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

SOURCES := \
    $(STFFT_ROOT)/R2FFT/hdl/bfp_Shifter.sv \
    $(STFFT_ROOT)/R2FFT/hdl/bfp_bitWidthAcc.sv \
    $(STFFT_ROOT)/R2FFT/hdl/bfp_bitWidthDetector.sv \
    $(STFFT_ROOT)/R2FFT/hdl/bfp_maxBitWidth.sv \
    $(STFFT_ROOT)/R2FFT/hdl/bitReverseCounter.sv \
    $(STFFT_ROOT)/R2FFT/hdl/ramPipelineBridge.sv \
    $(STFFT_ROOT)/R2FFT/hdl/radix2Butterfly.sv \
    $(STFFT_ROOT)/R2FFT/hdl/butterflyCore.sv \
    $(STFFT_ROOT)/R2FFT/hdl/twiddleFactorRomBridge.sv \
    $(STFFT_ROOT)/R2FFT/hdl/butterflyUnit.sv \
    $(STFFT_ROOT)/R2FFT/hdl/fftAddressGenerator.sv \
    $(STFFT_ROOT)/R2FFT/hdl/readBusMux.sv \
    $(STFFT_ROOT)/R2FFT/hdl/writeBusMux.sv \
    $(STFFT_ROOT)/R2FFT/hdl/R2FFT.sv \
    $(STFFT_ROOT)/tests/models/gf180mcu_fd_ip_sram__sram128x8m8wm1.v \
    $(STFFT_ROOT)/ZipCPU/ip/Window/windowfn.v \
    $(STFFT_ROOT)/rtl/fft_data_ram.sv \
    $(STFFT_ROOT)/rtl/fft_twiddle_rom.sv \
    $(STFFT_ROOT)/rtl/stfft.sv

COCOTB_MAKEFILES := $(shell cocotb-config --makefiles)

# Waveform generation (use VCD for better compatibility)
SIM_ARGS = --trace --trace-structs --trace-depth 2

export COCOTB_HDL_TIMEUNIT = 1ns
export COCOTB_HDL_TIMEPRECISION = 1ps

test-stfft:
	@echo "Creating hanning.hex symlink..."
	ln -sf $(STFFT_ROOT)/ZipCPU/hanning.hex hanning.hex
	@echo "Running simulation with waveform generation..."
	SIM=verilator \
	TOPLEVEL_LANG=verilog \
	TOPLEVEL=$(TOP) \
	MODULE=test_$(TOP) \
	COCOTB_TEST_MODULES=test_$(TOP) \
	EXTRA_ARGS="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal $(SIM_ARGS)" \
	VERILOG_SOURCES="$(SOURCES)" \
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim

test-debug:
	@echo "Creating hanning.hex symlink..."
	ln -sf $(STFFT_ROOT)/ZipCPU/hanning.hex hanning.hex
	@echo "Running debug test..."
	SIM=verilator \
	TOPLEVEL_LANG=verilog \
	TOPLEVEL=$(TOP) \
	MODULE=test_$(TOP) \
	COCOTB_TEST_MODULES=test_$(TOP) \
	EXTRA_ARGS="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal $(SIM_ARGS)" \
	VERILOG_SOURCES="$(SOURCES)" \
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim TESTCASE=test_debug_window

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log hanning.hex

.PHONY: test-stfft test-debug clean