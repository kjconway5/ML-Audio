#!/usr/bin/env python3
"""
Utility functions for Cocotb-based testing and simulation management

Each module directory must have filelist.json with keys for "top" and "files":
{
    "top": "module_name",
    "files": [
        "rtl/module_name.sv",
        "rtl/submodule.sv"
    ]
}

Each file in the filelist is relative to the repository root.
"""

import os
import sys
import json
from pathlib import Path

try:
    import git
    import cocotb
    from cocotb_test.simulator import run
    from cocotb.clock import Clock
    from cocotb.regression import TestFactory
    from cocotb.utils import get_sim_time
    from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, with_timeout
    from cocotb.types import LogicArray
except ImportError as e:
    print(f"Warning: Some imports failed: {e}", file=sys.stderr)
    print("Make sure cocotb and cocotb-test are installed:", file=sys.stderr)
    print("  pip install cocotb cocotb-test gitpython", file=sys.stderr)



# File and Path Management

def get_files_from_filelist(p, n="filelist.json"):
    """Get a list of files from a json filelist.
    
    Arguments:
        p -- Path to the directory that contains the .json file
        n -- name of the .json file to read.
    
    Returns:
        List of file paths (relative to repository root)
    """
    json_path = os.path.join(p, n)
    with open(json_path) as filelist:
        files = json.load(filelist)["files"]
    return files


def get_sources(r, p, jsonname="filelist.json"):
    """Get a list of absolute source file paths from a json filelist.
    
    Arguments:
        r -- Absolute path to the root of the repository.
        p -- Absolute path to the directory containing filelist.json
        jsonname -- Name of the JSON file (default: filelist.json)
    
    Returns:
        List of absolute file paths
    """
    sources = get_files_from_filelist(p, jsonname)
    sources = [os.path.join(r, f) for f in sources]
    return sources


def get_top_from_filelist(p, n="filelist.json"):
    """Get the name of the top level module from a json filelist.
    
    Arguments:
        p -- Absolute path to the directory containing filelist.json
        n -- name of the .json file to read.
    
    Returns:
        String name of top module
    """
    json_path = os.path.join(p, n)
    with open(json_path) as filelist:
        top = json.load(filelist)["top"]
        return top


def get_top(p, n="filelist.json"):
    """Get the name of the top level module from a filelist.json.
    
    Arguments:
        p -- Absolute path to the directory containing json filelist
        n -- Name of the json filelist, defaults to filelist.json
    
    Returns:
        String name of top module
    """
    return get_top_from_filelist(p, n)


def get_repo_root():
    """Get the repository root using git"""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.working_tree_dir
    except:
        # Fallback if not in a git repo
        return os.getcwd()


# Cocotb Test Runner

def runner(simulator, timescale, tbpath, params, defs=[], testname=None, 
           pymodule=None, jsonpath=None, jsonname="filelist.json", root=None):
    """Run the simulator on test n, with parameters params, and defines defs.
    
    Arguments:
        simulator -- Simulator to use ('icarus', 'verilator', etc.)
        timescale -- Timescale string (e.g., '1ns/1ps')
        tbpath -- Path to testbench directory
        params -- Dictionary of parameters to pass to the design
        defs -- List of defines
        testname -- Specific test to run (None = run all tests)
        pymodule -- Python test module name (default: test_<top>)
        jsonpath -- Path to directory containing filelist.json
        jsonname -- Name of JSON file (default: filelist.json)
        root -- Repository root (auto-detected if None)
    """
    
    # If json path is none, assume that it is the same as tbpath
    if jsonpath is None:
        jsonpath = tbpath
    
    assert os.path.exists(jsonpath), f"jsonpath directory must exist: {jsonpath}"
    
    top = get_top(jsonpath, jsonname)
    
    # If pymodule is none, assume that the python module name is test_<top>
    if pymodule is None:
        pymodule = "test_" + top
    
    # Determine test directory name
    if testname is None:
        testdir = "all"
    else:
        testdir = testname
    
    # Assume all paths in the json file are relative to the repository root.
    if root is None:
        root = get_repo_root()
    
    assert os.path.exists(root), f"root directory path must exist: {root}"
    
    sources = get_sources(root, jsonpath, jsonname)
    
    work_dir = os.path.join(tbpath, "run", testdir, get_param_string(params), simulator)
    build_dir = os.path.join(tbpath, "build", get_param_string(params))
    
    # Icarus doesn't build, it just runs.
    if simulator.startswith("icarus"):
        build_dir = work_dir
    
    # Verilator-specific settings
    if simulator.startswith("verilator"):
        compile_args = ["-Wno-fatal", "-DVM_TRACE_FST=1", "-DVM_TRACE=1"]
        plus_args = ["--trace", "--trace-fst"]
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
    else:
        compile_args = []
        plus_args = []
    
    run(verilog_sources=sources,
        simulator=simulator,
        toplevel=top,
        module=pymodule,
        compile_args=compile_args,
        plus_args=plus_args,
        sim_build=build_dir,
        timescale=timescale,
        parameters=params,
        defines=defs + ["VM_TRACE_FST=1", "VM_TRACE=1"],
        work_dir=work_dir,
        waves=True,
        testcase=testname)


# Linting Support

def lint(simulator, timescale, tbpath, params, defs=[], compile_args=[], 
         pymodule=None, jsonpath=None, jsonname="filelist.json", root=None):
    """Build (run) the lint and style checks.
    
    Arguments:
        simulator -- Simulator to use (typically 'verilator' for linting)
        timescale -- Timescale string
        tbpath -- Path to testbench directory
        params -- Dictionary of parameters
        defs -- List of defines
        compile_args -- Additional compilation arguments
        pymodule -- Python test module name
        jsonpath -- Path to directory containing filelist.json
        jsonname -- Name of JSON file
        root -- Repository root
    """
    
    if jsonpath is None:
        jsonpath = tbpath
    
    assert os.path.exists(jsonpath), f"jsonpath directory must exist: {jsonpath}"
    
    top = get_top(jsonpath, jsonname)
    
    if root is None:
        root = get_repo_root()
    
    assert os.path.exists(root), f"root directory path must exist: {root}"
    
    sources = get_sources(root, jsonpath, jsonname)
    
    if pymodule is None:
        pymodule = "test_" + top
    
    # Create the expected makefile so cocotb-test won't complain.
    sim_build = "lint"
    if not os.path.exists("lint"):
        os.mkdir("lint")
    
    with open("lint/Vtop.mk", 'w') as fd:
        fd.write("all:\n")
    
    make_args = ["-n"]
    compile_args += ["--lint-only"]
    
    run(verilog_sources=sources,
        simulator=simulator,
        toplevel=top,
        module=pymodule,
        compile_args=compile_args,
        sim_build=sim_build,
        timescale=timescale,
        parameters=params,
        defines=defs,
        make_args=make_args,
        compile_only=True)


# Parameter Utilities

def get_param_string(parameters):
    """Get a string of all the parameters concatenated together.
    
    Arguments:
        parameters -- a dictionary of key-value pairs
    
    Returns:
        String with format "key1=val1_key2=val2"
    """
    if not parameters:
        return "default"
    return "_".join(("{}={}".format(*i) for i in parameters.items()))



# Cocotb Helper Functions


def assert_resolvable(s):
    """Assert that a signal has no X or Z values"""
    assert s.value.is_resolvable, \
        f"Unresolvable value in {s._path} (x or z in some or all bits) at Time {get_sim_time(units='ns')}ns."


def assert_passerror(s):
    """Assert that pass/error signal is explicitly set"""
    assert s.value.is_resolvable, \
        f"Testbench pass/fail output ({s._path}) is set to x or z, but must be explicitly set to 0 at start of simulation."


async def clock_start_sequence(clk_i, period=1, unit='ns'):
    """Start a clock with proper initialization sequence.
    
    Arguments:
        clk_i -- Clock signal
        period -- Clock period (default: 1)
        unit -- Time unit (default: 'ns')
    """
    # Set the clock to Z for 10 ns. This helps separate tests.
    clk_i.value = LogicArray(['z'])
    await Timer(10, 'ns')
    
    # Create clock object
    c = Clock(clk_i, period, unit)
    
    # Start the clock (soon). Start it low to avoid issues on the first RisingEdge
    cocotb.start_soon(c.start(start_high=False))


async def reset_sequence(clk_i, reset_i, cycles, FinishClkFalling=True, active_level=True):
    """Perform a reset sequence.
    
    Arguments:
        clk_i -- Clock signal
        reset_i -- Reset signal
        cycles -- Number of clock cycles to hold reset
        FinishClkFalling -- If True, finish on falling edge (default: True)
        active_level -- Active level of reset (True=active high, default: True)
    """
    reset_i.setimmediatevalue(not active_level)
    
    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = active_level
    
    await ClockCycles(clk_i, cycles)
    
    # Always assign inputs on the falling edge
    await FallingEdge(clk_i)
    reset_i.value = not active_level
    
    reset_i._log.debug("Reset complete")
    
    # Always assign inputs on the falling edge
    if not FinishClkFalling:
        await RisingEdge(clk_i)


async def delay_cycles(dut, ncyc, polarity):
    """Wait for a number of clock cycles.
    
    Arguments:
        dut -- Device under test
        ncyc -- Number of cycles to wait
        polarity -- If True, wait for rising edges; if False, wait for falling edges
    """
    for _ in range(ncyc):
        if polarity:
            await RisingEdge(dut.clk_i)
        else:
            await FallingEdge(dut.clk_i)

