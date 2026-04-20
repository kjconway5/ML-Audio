#!/usr/bin/env python3
"""Conftest for cocotb-test pytest integration"""

import sys
from pathlib import Path
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "util"))

collect_ignore = []