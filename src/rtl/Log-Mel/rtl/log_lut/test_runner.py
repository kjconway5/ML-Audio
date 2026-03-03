#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import os
import sys
import pytest
from pathlib import Path

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "util"))

from utilities import runner, get_repo_root

COCOTB_TESTS = [
    "test_log_lut_basic_writes",
    "test_log_lut_full_sweep",
]

def _stage_hex_into_workdir(hex_src: Path, work_dir: Path):
    """
    Symlink the LUT hex into the simulator work dir so $readmemh('log2_lut.hex') can find it
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    assert hex_src.exists(), f"Missing LUT file: {hex_src}"

    link = work_dir / hex_src.name
    if link.exists():
        return

    os.symlink(hex_src.resolve(), link)

@pytest.mark.parametrize("testname", COCOTB_TESTS)
def test_log_lut_cocotb(testname):
    """
    Run cocotb tests for log_lut
    """
    test_dir = Path(__file__).parent

    # This is the *actual* work_dir used when runner(testname=...) is set
    work_dir = test_dir / "run" / testname / "default" / "icarus"

    # Your RTL does: $readmemh("log2_lut.hex")
    hex_src = test_dir / "log2_lut.hex"
    _stage_hex_into_workdir(hex_src, work_dir)

    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=testname,
        pymodule="test_log_lut",
        jsonpath=str(test_dir),
        root=get_repo_root(),
    )