# Top-level Makefile for ASIC Design Project
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || echo .)
RTL_DIR := $(REPO_ROOT)/rtl
MODULES := $(notdir $(wildcard $(RTL_DIR)/*))

.DEFAULT_GOAL := help

# Helper to run make on all modules
define run_on_modules
	@for m in $(MODULES); do \
		[ -f "$(RTL_DIR)/$$m/Makefile" ] && $(MAKE) -s -C $(RTL_DIR)/$$m $(1) || true; \
	done
endef

# Project-wide operations
test-all:
	$(call run_on_modules,test-sv)
	$(call run_on_modules,test-cocotb)

lint-all:
	$(call run_on_modules,lint)

synth-all:
	$(call run_on_modules,synth-abstract)

clean-all:
	$(call run_on_modules,clean)

list-modules:
	@printf "Modules: %s\n" "$(MODULES)"

tree:
	@tree -L 3 -I '__pycache__|*.pyc|run|build|lint|.git' 2>/dev/null || \
		find . -maxdepth 3 -type d ! -path '*/\.*' ! -path '*/run' ! -path '*/build' ! -path '*/__pycache__' | sort

new-module:
	@[ -n "$(NAME)" ] || { echo "Usage: make new-module NAME=<name>"; exit 1; }
	@bash $(REPO_ROOT)/scripts/new_module.sh $(NAME)

# Environment check
define check_cmd
	@command -v $(1) >/dev/null 2>&1 && printf "  ✓ %s\n" "$(1)" || printf "  ✗ %s\n" "$(1)"
endef

define check_py
	@python3 -c "import $(1)" 2>/dev/null && printf "  ✓ %s\n" "$(1)" || printf "  ✗ %s\n" "$(1)"
endef

check-env:
	@echo "Required Tools:"
	$(call check_cmd,iverilog)
	$(call check_cmd,verilator)
	$(call check_cmd,yosys)
	$(call check_cmd,gtkwave)
	$(call check_cmd,python3)
	@echo "Python Packages:"
	$(call check_py,cocotb)
	$(call check_py,cocotb_test)
	$(call check_py,pytest)

install-deps:
	pip install cocotb cocotb-test pytest pytest-json-report gitpython

# Docker operations
docker-build:      ; docker-compose build
docker-shell:      ; docker-compose run --rm rtl-dev
docker-test:       ; docker-compose run --rm test
docker-lint:       ; docker-compose run --rm lint
docker-synth:      ; docker-compose run --rm synth
docker-check-env:  ; docker-compose run --rm rtl-dev make check-env
docker-clean:      ; docker-compose down --rmi local --volumes --remove-orphans

# Help
help:
	@echo "Design Project"
	@echo ""
	@echo "Project:       test-all | lint-all | synth-all | clean-all | list-modules"
	@echo "Setup:         new-module NAME=x | check-env | install-deps | tree"
	@echo "Docker:        docker-build | docker-shell | docker-test | docker-lint | docker-synth | docker-clean"
	@echo ""
	@echo "Module usage:  cd rtl/<module> && make [test-sv|test-cocotb|synth-abstract|help]"
	@echo "Modules:       $(MODULES)"

quickref: ; @cat $(REPO_ROOT)/QUICKREF.md
readme:   ; @cat $(REPO_ROOT)/README.md

.PHONY: help test-all lint-all synth-all clean-all list-modules tree new-module \
        check-env install-deps quickref readme docker-build docker-shell \
        docker-test docker-lint docker-synth docker-check-env docker-clean