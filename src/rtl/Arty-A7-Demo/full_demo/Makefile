TOP   := full_demo_top
BIT   := build/$(TOP).bit
PORT  ?= /dev/ttyUSB1
BAUD  ?= 460800
BOARD := arty_a7_100t

# Trained model dir: provides weights.hex, bias.hex, scales.txt;
# cfg.hex generated below from scales.txt.
MODEL_DIR ?= ../../../ml/models/dscnn-24center-v1
WEIGHTS   := $(MODEL_DIR)/weights.hex
BIAS      := $(MODEL_DIR)/bias.hex
SCALES    := $(MODEL_DIR)/scales.txt
CFG       := host/cfg.hex

GEN_CFG   := host/gen_cfg_hex.py

.PHONY: build flash listen probe boot stream classify demo cfg clean

# `make demo` config — best model + matching cfg per session validation
# (full-v2 wins on live mic over center-v1 because of random-alignment
# training augmentation). Override any of these on the command line, e.g.
# `make demo DEMO_MODEL_DIR=../../../ml/models/dscnn-24center-v1 DEMO_CFG=host/cfg.hex`.
DEMO_MODEL_DIR ?= ../../../ml/models/dscnn-24full-v2
DEMO_WEIGHTS   := $(DEMO_MODEL_DIR)/weights.hex
DEMO_BIAS      := $(DEMO_MODEL_DIR)/bias.hex
DEMO_CFG       ?= host/cfg-24full-v2.hex
DEMO_VAD       ?= off       # off | auto | <int>
DEMO_SMOOTH    ?= 1         # ignored under --mic-verbose; bump for dedup runs

build: $(BIT)

$(BIT): full_demo_top.sv class_reporter.sv arty_full.xdc build.tcl
	vivado -mode batch -source build.tcl

flash: $(BIT)
	openFPGALoader -b $(BOARD) "$(BIT)"

cfg: $(CFG)

$(CFG): $(SCALES) $(GEN_CFG)
	python3 $(GEN_CFG) $(SCALES) -o $(CFG)

# Boot all SRAMs (features + DSCNN) and assert boot_done. After this,
# kws_top.start goes high and inference fires on every spect_done.
boot: $(CFG)
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --boot \
	        --weights $(WEIGHTS) --bias $(BIAS) --cfg $(CFG)

# End-to-end demo: boot + stream a synthetic 1 kHz sine + listen for
# class result tagged byte from class_reporter.
stream: $(CFG)
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --stream \
	        --weights $(WEIGHTS) --bias $(BIAS) --cfg $(CFG)

# Same as `stream` but uses a WAV file on disk as the audio source.
classify: $(CFG)
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --classify $(WAV) \
	        --weights $(WEIGHTS) --bias $(BIAS) --cfg $(CFG)

# Live-mic demo: flash the bitstream, then run --mic-continuous in
# verbose (firehose) mode so the class stream looks live instead of
# collapsing into dedup ×N lines. Speak yes/no/on/off/wow into the
# laptop mic; classes scroll at ~20 Hz. Ctrl-C to stop.
#
# Does NOT trigger Vivado — fails if $(BIT) is missing.
demo: $(DEMO_CFG)
	@if [ ! -f "$(BIT)" ]; then \
	    echo "ERROR: $(BIT) not found. 'make build' on a Vivado machine, or"; \
	    echo "       drop a pre-built .bit at $(BIT)."; \
	    exit 1; \
	fi
	@if [ ! -f "$(DEMO_WEIGHTS)" ] || [ ! -f "$(DEMO_BIAS)" ]; then \
	    echo "ERROR: demo model files missing under $(DEMO_MODEL_DIR)/"; \
	    echo "       (need weights.hex and bias.hex)"; \
	    exit 1; \
	fi
	@echo "=== demo: flashing $(BIT) ==="
	openFPGALoader -b $(BOARD) "$(BIT)"
	@echo
	@echo "=== demo: live mic ==="
	@echo "  model:  $(DEMO_MODEL_DIR)"
	@echo "  cfg:    $(DEMO_CFG)"
	@echo "  vad:    $(DEMO_VAD)    smoothing: $(DEMO_SMOOTH)"
	@echo "  speak:  yes / no / on / off / wow"
	@echo "  Ctrl-C to stop."
	@echo
	python3 -u host/load_full.py -p $(PORT) -b $(BAUD) --mic-continuous \
	        --weights $(DEMO_WEIGHTS) --bias $(DEMO_BIAS) --cfg $(DEMO_CFG) \
	        --vad $(DEMO_VAD) --smoothing $(DEMO_SMOOTH) --mic-verbose

# `make demo-cfg MODEL=<dir>` — regenerate a cfg-<model>.hex matched to
# whatever model you point DEMO_MODEL_DIR at. Run once after switching
# checkpoints; cfg-24full-v2.hex is already in tree for the default.
$(DEMO_CFG): $(DEMO_MODEL_DIR)/scales.txt $(GEN_CFG)
	python3 $(GEN_CFG) $(DEMO_MODEL_DIR)/scales.txt -o $(DEMO_CFG)

listen:
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --listen-only

probe:
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --probe

clean:
	rm -rf build *.jou *.log .Xil host/cfg.hex
