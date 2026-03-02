#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import os
import sys
from pathlib import Path

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "util"))

from utilities import runner, get_repo_root


def test_frame_control_cocotb():
    test_dir = Path(__file__).parent
    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=None,
        pymodule="test_frame_control",
        jsonpath=str(test_dir),
        jsonname="filelist.json",
        root=get_repo_root()
    )