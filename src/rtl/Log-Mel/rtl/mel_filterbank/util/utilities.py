import os
import json
from pathlib import Path
from cocotb_test.simulator import run



# Repo / filelist handling
def get_repo_root():
    # /workspace/src/rtl (NOT Log-Mel)
    return Path("/workspace")

def get_top(json_path):
    with open(os.path.join(json_path, "filelist.json")) as f:
        return json.load(f)["top"]


def get_sources(root, json_path):
    with open(os.path.join(json_path, "filelist.json")) as f:
        files = json.load(f)["files"]

    return [str(Path(root) / f) for f in files]


def runner(simulator, timescale, tbpath, jsonpath,
           pymodule, testname=None, root=None):

    if root is None:
        root = get_repo_root()

    top = get_top(jsonpath)
    sources = get_sources(root, jsonpath)

    work_dir = os.path.join(tbpath, "run")

    run(
        verilog_sources=sources,
        toplevel=top,
        module=pymodule,
        simulator=simulator,
        timescale=timescale,
        sim_build=work_dir,
        waves=True,
        testcase=testname
    )