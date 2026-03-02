#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import sys
from pathlib import Path

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "util"))

from utilities import runner, get_repo_root


def test_window_cocotb():
    test_dir = Path(__file__).parent
    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={
            "LGNFFT": 8,
            "IW": 14,
            "OW": 14,
            "TW": 14,
        },
        defs=[],
        testname=None,
        pymodule="test_window",
        jsonpath=str(test_dir),
        root=get_repo_root()
    )
