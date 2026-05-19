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

.PHONY: build flash listen probe boot stream classify cfg clean

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

listen:
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --listen-only

probe:
	python3 host/load_full.py -p $(PORT) -b $(BAUD) --probe

clean:
	rm -rf build *.jou *.log .Xil host/cfg.hex
