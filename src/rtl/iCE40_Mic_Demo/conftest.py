import os

def pytest_configure(config):
    os.environ["COCOTB_REDUCED_LOG_FMT"] = "1"