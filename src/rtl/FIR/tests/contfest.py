#!/usr/bin/env python3
"""Conftest for cocotb-test pytest integration."""
import os
import sys
from pathlib import Path
 
os.environ['PYTHONIOENCODING'] = 'utf-8'
 
# Add shared util directory to path if it exists
_UTIL_DIR = Path(__file__).resolve().parent.parent.parent / "util"
if _UTIL_DIR.is_dir():
    sys.path.insert(0, str(_UTIL_DIR))
 
# Prevent pytest from collecting non-test files
collect_ignore = []
 
