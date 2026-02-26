#!/usr/bin/env python3
"""Pytest runner for cocotb tests"""

import os
import sys
from pathlib import Path

# Add util to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "util"))

from utilities import runner, get_repo_root


def _link_hex_files(src_dir, dst_dir):
    """Symlink .hex files into the simulation run directory so $readmemh can find them."""
    os.makedirs(dst_dir, exist_ok=True)
    for hex_file in Path(src_dir).glob("*.hex"):
        link = Path(dst_dir) / hex_file.name
        if not link.exists():
            os.symlink(hex_file.resolve(), link)


def test_mel_filterbank_cocotb():
    """Run cocotb tests for mel_filterbank."""
    test_dir = Path(__file__).parent
    run_dir = test_dir / "run" / "all" / "default" / "icarus"

    # mel_coeff_rom uses $readmemh("../data/...") relative to the sim working dir.
    # The sim runs from run_dir, so "../data/" resolves to run_dir/../data/.
    # Symlink hex files there so $readmemh can find them.
    data_dir = test_dir.parent.parent / "data"
    target_data_dir = run_dir.parent / "data"
    _link_hex_files(data_dir, target_data_dir)

    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        params={},
        defs=[],
        testname=None,
        pymodule="test_mel_filterbank",
        jsonpath=str(test_dir),
        root=get_repo_root()
    )
