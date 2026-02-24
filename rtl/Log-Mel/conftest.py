#!/usr/bin/env python3
"""Conftest for cocotb-test pytest integration"""

import sys
from pathlib import Path
import logging
import cocotb.log

# make cocotb logs show in terminal
cocotb.log.default_config()
logging.getLogger("cocotb").setLevel(logging.DEBUG)
logging.getLogger("cocotb").handlers = []
logging.getLogger("cocotb").addHandler(logging.StreamHandler())
logging.getLogger("cocotb").propagate = False

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "util"))

collect_ignore = ["test_example.py"]