

# Path to the repository root
REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

# Tool paths (override if installed in non-standard locations)
YOSYS ?= yosys
NETLISTSVG ?= netlistsvg
RSVG ?= rsvg-convert
PYTHON ?= python3

# Read source files and top module from filelist.json
SYNTH_SOURCES = $(shell $(PYTHON) $(REPO_ROOT)/util/get_filelist.py)
SYNTH_SOURCES := $(addprefix $(REPO_ROOT)/,$(SYNTH_SOURCES))
SYNTH_TOP = $(shell $(PYTHON) $(REPO_ROOT)/util/get_top.py)


TECH ?= sky130

all: help

# Abstract Synthesis (Technology-Independent)

synth-abstract: abstract.json abstract.pdf
	@echo "Abstract synthesis complete."

abstract.json: filelist.json $(SYNTH_SOURCES)
	@echo "Synthesizing to abstract boolean gates..."
	$(YOSYS) -ql abstract.yslog -p 'prep -top $(SYNTH_TOP); flatten; json -o $@' $(SYNTH_SOURCES)

abstract.pdf: abstract.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@


# Generic Synthesis (Technology-Mapped Generic Library)


synth-generic: generic.json generic.pdf
	@echo "Generic synthesis complete."

generic.json: filelist.json $(SYNTH_SOURCES)
	@echo "Synthesizing to generic library..."
	$(YOSYS) -ql generic.yslog -p 'synth -top $(SYNTH_TOP); json -o $@' $(SYNTH_SOURCES)

generic.pdf: generic.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@

#==============================================================================
# Technology-Specific Synthesis
#==============================================================================

# SkyWater 130nm synthesis
synth-sky130: sky130.json
	@echo "SkyWater 130nm synthesis complete."

sky130.json: filelist.json $(SYNTH_SOURCES)
	@echo "Synthesizing for SkyWater 130nm..."
	$(YOSYS) -ql sky130.yslog -p 'synth -top $(SYNTH_TOP); json -o $@' $(SYNTH_SOURCES)

sky130.pdf: sky130.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@


# FPGA Synthesis 

synth-fpga: fpga.json
	@echo "FPGA synthesis complete."

fpga.json: filelist.json $(SYNTH_SOURCES)
	@echo "Synthesizing for FPGA..."
	$(YOSYS) -ql fpga.yslog -p 'synth_ice40 -top $(SYNTH_TOP) -json $@' $(SYNTH_SOURCES)

fpga.pdf: fpga.json
	$(NETLISTSVG) $< -o $(subst pdf,svg,$@)
	$(RSVG) -f pdf $(subst pdf,svg,$@) -o $@


# Gate Count and Resource Analysis


stats: filelist.json $(SYNTH_SOURCES)
	@echo "==================================================================="
	@echo "Resource Statistics for $(SYNTH_TOP)"
	@echo "==================================================================="
	$(YOSYS) -p 'read_verilog $(SYNTH_SOURCES); hierarchy -check -top $(SYNTH_TOP); \
	             proc; opt; memory; opt; fsm; opt; techmap; opt; \
	             stat -width -liberty' 2>&1 | tee stats.log
	@echo ""
	@echo "Detailed statistics saved to stats.log"

# Extract key metrics and compare to constraints
analyze: stats
	@echo ""
	@echo "==================================================================="
	@echo "ASIC Resource Analysis vs. Constraints"
	@echo "==================================================================="
	@echo "Target Core Area:    12.92 mm²"
	@echo "NAND2 Gate Area:     10.976 μm²"
	@echo "50% Cell Usage:      588,556 gates (target)"
	@echo "100% Cell Usage:     1,177,113 gates (max)"
	@echo "==================================================================="
	@grep -E "(Number of cells|area)" stats.log || echo "Parse stats.log manually for gate count"


# Cleanup Targets


synth-clean:
	rm -rf abstract.json abstract.yslog abstract.svg abstract.pdf
	rm -rf generic.json generic.yslog generic.svg generic.pdf
	rm -rf sky130.json sky130.yslog sky130.svg sky130.pdf
	rm -rf fpga.json fpga.yslog fpga.svg fpga.pdf
	rm -rf stats.log
	rm -rf *.dot

synth-extraclean: synth-clean
	rm -rf *.rpt


# Help Targets


synth-help:
	@echo ""
	@echo "Synthesis Targets:"
	@echo "  synth-abstract  : Synthesize to abstract boolean gates (tech-independent)"
	@echo "  abstract.pdf    : Generate PDF schematic of abstract synthesis"
	@echo "  synth-generic   : Synthesize to generic library cells"
	@echo "  generic.pdf     : Generate PDF schematic of generic synthesis"
	@echo "  synth-sky130    : Synthesize for SkyWater 130nm process"
	@echo "  sky130.pdf      : Generate PDF schematic for SkyWater 130nm"
	@echo "  synth-fpga      : Synthesize for FPGA (prototyping)"
	@echo "  fpga.pdf        : Generate PDF schematic for FPGA"
	@echo "  stats           : Generate detailed resource statistics"
	@echo "  analyze         : Analyze resources vs. ASIC constraints"
	@echo "  synth-clean     : Remove all synthesis outputs"
	@echo "  synth-extraclean: Remove all generated files (includes synth-clean)"

synth-vars-help:
	@echo ""
	@echo "Synthesis Variables (override in environment or command line):"
	@echo "  YOSYS       : Path to yosys executable"
	@echo "  NETLISTSVG  : Path to netlistsvg executable"
	@echo "  RSVG        : Path to rsvg-convert executable"
	@echo "  TECH        : Target technology (default: sky130)"

clean: synth-clean
targets-help: synth-help
vars-help: synth-vars-help

.PHONY: all synth-abstract synth-generic synth-sky130 synth-fpga stats analyze \
        synth-clean synth-extraclean synth-help synth-vars-help \
        clean targets-help vars-help