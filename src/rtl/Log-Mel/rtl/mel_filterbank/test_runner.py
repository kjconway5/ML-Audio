#!/usr/bin/env python3
import os
from pathlib import Path

from util.utilities import runner, get_repo_root


# ----------------------------
# Stage HEX files into sim dir
# ----------------------------
def stage_hex_files(run_dir: Path):
    logmel_root = Path("/workspace/src/rtl/Log-Mel")
    data_dir = logmel_root / "data"

    run_dir.mkdir(exist_ok=True, parents=True)

    for f in data_dir.glob("*.hex"):
        dst = run_dir / f.name

        if dst.exists():
            dst.unlink()

        dst.symlink_to(f)


# ----------------------------
# Main cocotb entry test
# ----------------------------
def test_mel_filterbank_new_cocotb():
    test_dir = Path(__file__).parent
    run_dir = test_dir / "run"

    stage_hex_files(run_dir)

    root = get_repo_root()

    runner(
        simulator="icarus",
        timescale="1ns/1ps",
        tbpath=str(test_dir),
        jsonpath=str(test_dir),
        pymodule="test_mel_filterbank",
        testname=None,
        root=root
    )