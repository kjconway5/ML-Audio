#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import pytest
from pathlib import Path
from utilities import runner, get_repo_root

COCOTB_TESTS = [
    "test_frame_control_fsm",
    "test_melindex",
]

@pytest.mark.parametrize("testname", COCOTB_TESTS)
def test_frame_control_cocotb(testname):
    test_dir = Path(__file__).parent
    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=testname,          # run ONLY this cocotb test
        pymodule="test_frame_control",
        jsonpath=str(test_dir),
        jsonname="filelist.json",
        root=get_repo_root()
    )