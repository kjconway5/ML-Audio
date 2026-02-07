# Top-level Makefile
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






.PHONY: help test-all lint-all synth-all clean-all list-modules tree new-module \
        