#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import sys
from pathlib import Path

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "util"))
from utilities import runner, get_repo_root

def test_power_calc_cocotb():
    test_dir = Path(__file__).parent
    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=None,
        pymodule="test_power_calc",
        jsonpath=str(test_dir),
        jsonname="filelist_power_calc.json",
        root=get_repo_root()
    )

def test_mac_unit_cocotb():
    test_dir = Path(__file__).parent
    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=None,
        pymodule="test_mac_unit",
        jsonpath=str(test_dir),
        jsonname="filelist_mac_unit.json",
        root=get_repo_root()
    )

def test_logmel_top_cocotb():
    test_dir = Path(__file__).parent
    run_dir  = test_dir / "run" / "all" / "default" / "icarus"
    data_dir = test_dir / "data"

    # symlink hex files into run dir so $readmemh can find them
    run_dir.mkdir(parents=True, exist_ok=True)
    for hex_file in data_dir.glob("*.hex"):
        link = run_dir / hex_file.name
        if not link.exists():
            link.symlink_to(hex_file.resolve())

    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=None,
        pymodule="test_logmel_top",
        jsonpath=str(test_dir),
        jsonname="filelist_logmel_top.json",
        root=get_repo_root()
    )
