REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

-include $(REPO_ROOT)/frag/simulate.mk


.DEFAULT_GOAL := help

# Cocotb / pytest targets
PYTEST ?= pytest
PYTEST_ARGS ?= -s
TEST ?= test_log_lut_cocotb
SIM ?= icarus
WAVES ?= 1

test:
	@echo "Running all pytest wrappers"
	SIM=$(SIM) WAVES=$(WAVES) COCOTB_LOG_LEVEL=INFO \
		pytest -s

sim: test 
.PHONY: test sim