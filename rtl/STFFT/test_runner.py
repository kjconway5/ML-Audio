#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import os
import sys
from pathlib import Path

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "util"))

from utilities import runner, get_repo_root


def _link_hex_files(src_dir, dst_dir):
    """Symlink .hex files into the simulation run directory so $readmemh can find them."""
    os.makedirs(dst_dir, exist_ok=True)
    for hex_file in Path(src_dir).glob("*.hex"):
        link = Path(dst_dir) / hex_file.name
        if not link.exists():
            os.symlink(hex_file.resolve(), link)


def test_example_cocotb():
    """Run cocotb tests """
    test_dir = Path(__file__).parent
    run_dir = test_dir / "run" / "all" / "default" / "icarus"

    # FFT modules use $readmemh with bare filenames — symlink hex files into run dir
    _link_hex_files(test_dir / "FFT", run_dir)

    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=None,
        pymodule="test_stfft",
        jsonpath=str(test_dir),
        root=get_repo_root()
    )
