# Synthesis fragment
REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
YOSYS ?= yosys
NETLISTSVG ?= netlistsvg
RSVG ?= rsvg-convert
PYTHON ?= python3

SYNTH_SOURCES = $(addprefix $(REPO_ROOT)/,$(shell $(PYTHON) $(REPO_ROOT)/util/get_filelist.py))
SYNTH_TOP = $(shell $(PYTHON) $(REPO_ROOT)/util/get_top.py)

# PDF generation helper
define make_pdf
	$(NETLISTSVG) $(1).json -o $(1).svg
	$(RSVG) -f pdf $(1).svg -o $(1).pdf
endef

# Abstract synthesis
synth-abstract: abstract.json abstract.pdf
abstract.json: filelist.json $(SYNTH_SOURCES)
	$(YOSYS) -ql abstract.yslog -p 'prep -top $(SYNTH_TOP); flatten; json -o $@' $(SYNTH_SOURCES)
abstract.pdf: abstract.json ; $(call make_pdf,abstract)

# Generic synthesis
synth-generic: generic.json generic.pdf
generic.json: filelist.json $(SYNTH_SOURCES)
	$(YOSYS) -ql generic.yslog -p 'synth -top $(SYNTH_TOP); json -o $@' $(SYNTH_SOURCES)
generic.pdf: generic.json ; $(call make_pdf,generic)

# SkyWater 130nm
synth-sky130: sky130.json
sky130.json: filelist.json $(SYNTH_SOURCES)
	$(YOSYS) -ql sky130.yslog -p 'synth -top $(SYNTH_TOP); json -o $@' $(SYNTH_SOURCES)
sky130.pdf: sky130.json ; $(call make_pdf,sky130)

# FPGA (iCE40)
synth-fpga: fpga.json
fpga.json: filelist.json $(SYNTH_SOURCES)
	$(YOSYS) -ql fpga.yslog -p 'synth_ice40 -top $(SYNTH_TOP) -json $@' $(SYNTH_SOURCES)
fpga.pdf: fpga.json ; $(call make_pdf,fpga)

# Statistics
stats: filelist.json $(SYNTH_SOURCES)
	$(YOSYS) -p 'read_verilog $(SYNTH_SOURCES); hierarchy -check -top $(SYNTH_TOP); \
	             proc; opt; memory; opt; fsm; opt; techmap; opt; stat -width -liberty' | tee stats.log

analyze: stats
	@grep -E "(Number of cells|area)" stats.log || true

# Cleanup
synth-clean:
	rm -rf {abstract,generic,sky130,fpga}.{json,yslog,svg,pdf} stats.log *.dot

clean: synth-clean

# Help
help:
	@echo "Synthesis: synth-abstract | synth-generic | synth-sky130 | synth-fpga"
	@echo "Analysis:  stats | analyze"
	@echo "Cleanup:   synth-clean"

.PHONY: synth-abstract synth-generic synth-sky130 synth-fpga stats analyze synth-clean clean help