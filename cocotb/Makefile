REPO_ROOT := $(abspath ..)
SRC       := $(REPO_ROOT)/src
RTL       := $(SRC)/rtl
LOGMEL_DATA := $(RTL)/Log-Mel/data
HANNING_HEX := $(RTL)/STFFT/ZipCPU/hanning.hex

READMEMH_HEX_FILES := \
	$(HANNING_HEX) \
	$(LOGMEL_DATA)/mel_coeffs_sparse.hex \
	$(LOGMEL_DATA)/mel_indices.hex \
	$(LOGMEL_DATA)/log2_lut.hex \
	$(RTL)/dscnn/bias_SRAM/bias.hex

TOP    := chip_core
MODULE := chip_core_tb

SLOT   ?= 1x1

# Slot define: 1x1 → SLOT_1X1, 0p5x1 → SLOT_0P5X1, etc.
SLOT_UPPER := $(shell echo $(SLOT) | tr 'a-z' 'A-Z' | sed 's/\./_/g; s/-/_/g')

# boot_pkg.sv must be first so Icarus can resolve the package before chip_core.sv
SIM_SOURCES := \
	$(RTL)/flash/boot_pkg.sv \
	$(SRC)/chip_core.sv \
	$(RTL)/top/full_pipeline_top.sv \
	$(RTL)/top/pipeline_top.sv \
	$(RTL)/dscnn/kws_top.sv \
	$(RTL)/dscnn/bias_SRAM/bias_SRAM.sv \
	$(RTL)/dscnn/feature_sram/feature_sram.sv \
	$(RTL)/dscnn/fsm/FSM.sv \
	$(RTL)/dscnn/mac_array/mac_array.sv \
	$(RTL)/dscnn/requant/requant.sv \
	$(RTL)/dscnn/spectrogram_sram/spectrogram_sram.sv \
	$(RTL)/dscnn/weight_sram/weight_sram.sv \
	$(RTL)/Log-Mel/rtl/log_top/logmel_top.sv \
	$(RTL)/Log-Mel/rtl/frame_control/frame_control.sv \
	$(RTL)/Log-Mel/rtl/log_lut/log_lut_sram.sv \
	$(RTL)/Log-Mel/rtl/log_lut/log_lut.sv \
	$(RTL)/Log-Mel/rtl/mac_unit/mac_unit.sv \
	$(RTL)/Log-Mel/rtl/mel_filterbank/mel_coeff_sram.sv \
	$(RTL)/Log-Mel/rtl/mel_filterbank/mel_filterbank.sv \
	$(RTL)/Log-Mel/rtl/output_buffer/output_buffer.sv \
	$(RTL)/Log-Mel/rtl/power_calc/power_calc.sv \
	$(RTL)/Log-Mel/rtl/spectral_vad/spectral_vad.sv \
	$(RTL)/Log-Mel/ip/Add.sv \
	$(RTL)/Log-Mel/ip/AddMopCsv.sv \
	$(RTL)/Log-Mel/ip/Cpr.sv \
	$(RTL)/Log-Mel/ip/Encode.sv \
	$(RTL)/Log-Mel/ip/FullAdder.sv \
	$(RTL)/Log-Mel/ip/LeadZeroDet.sv \
	$(RTL)/Log-Mel/ip/Log2.sv \
	$(RTL)/Log-Mel/ip/MulPPGenUns.sv \
	$(RTL)/Log-Mel/ip/MulUns.sv \
	$(RTL)/Log-Mel/ip/PrefixAnd.sv \
	$(RTL)/Log-Mel/ip/PrefixAndOr.sv \
	$(RTL)/Log-Mel/ip/SqrPPGenSgn.sv \
	$(RTL)/Log-Mel/ip/SqrSgn.sv \
	$(RTL)/STFFT/rtl/stfft.sv \
	$(RTL)/STFFT/rtl/fft_twiddle_rom.sv \
	$(RTL)/STFFT/rtl/fft_data_ram.sv \
	$(RTL)/STFFT/R2FFT/hdl/R2FFT.sv \
	$(RTL)/STFFT/R2FFT/hdl/fft.sv \
	$(RTL)/STFFT/R2FFT/hdl/bfp_bitWidthAcc.sv \
	$(RTL)/STFFT/R2FFT/hdl/bfp_bitWidthDetector.sv \
	$(RTL)/STFFT/R2FFT/hdl/bfp_maxBitWidth.sv \
	$(RTL)/STFFT/R2FFT/hdl/bfp_Shifter.sv \
	$(RTL)/STFFT/R2FFT/hdl/bitReverseCounter.sv \
	$(RTL)/STFFT/R2FFT/hdl/butterflyCore.sv \
	$(RTL)/STFFT/R2FFT/hdl/butterflyUnit.sv \
	$(RTL)/STFFT/R2FFT/hdl/fftAddressGenerator.sv \
	$(RTL)/STFFT/R2FFT/hdl/radix2Butterfly.sv \
	$(RTL)/STFFT/R2FFT/hdl/ramPipelineBridge.sv \
	$(RTL)/STFFT/R2FFT/hdl/rwBusMux.sv \
	$(RTL)/STFFT/R2FFT/hdl/twiddleFactorRomBridge.sv \
	$(RTL)/flash/boot_controller.sv \
	$(RTL)/flash/features_boot_router.sv \
	$(RTL)/flash/dscnn_boot_router.sv \
	$(RTL)/flash/uart_rx.v \
	$(RTL)/flash/uart_tx.v \
	$(RTL)/flash/uart.v \
	$(RTL)/CIC/cic.sv \
	$(RTL)/SPECT_BUFFER/spect_buffer_ctrl.sv \
	$(RTL)/FIR/rtl/compFIR.sv

COCOTB_MAKEFILES := $(shell cocotb-config --makefiles)

link-hex:
	@for f in $(READMEMH_HEX_FILES); do ln -sf $$f . 2>/dev/null || true; done
.PHONY: link-hex

sim-core: link-hex
	KWS_KEYWORD="$(KWS_KEYWORD)" KWS_SAMPLE_INDEX="$(KWS_SAMPLE_INDEX)" KWS_SAMPLE_MATCH="$(KWS_SAMPLE_MATCH)" KWS_MANIFEST_JSON="$(KWS_MANIFEST_JSON)" \
		SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(MODULE) \
		VERILOG_SOURCES="$(abspath $(SIM_SOURCES))" \
		COMPILE_ARGS="-DSIM -DSLOT_$(SLOT_UPPER) -I$(abspath $(SRC)) -Pchip_core.NUM_INPUT_PADS=12 -Pchip_core.NUM_BIDIR_PADS=40 -Pchip_core.NUM_ANALOG_PADS=2" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: sim-core

clean:
	rm -rf sim_build __pycache__ results.xml *.vcd *.fst *.log hanning.hex mel_coeffs_sparse.hex mel_indices.hex log2_lut.hex
.PHONY: clean
