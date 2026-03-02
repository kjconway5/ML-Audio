import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

# ── Parameters matching RTL ────────────────────────────────────────
N_MELS = 40
OUT_W  = 16

# ── Helpers ───────────────────────────────────────────────────────

async def reset_dut(dut):
    dut.reset.value       = 1
    dut.load_i.value      = 0
    dut.cnn_ready_i.value = 0
    for i in range(N_MELS):
        dut.log_out_i[i].value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

async def load_frame(dut, values):
    """Simulate log_lut firing log_done — pulse load_i for one cycle with all 40 values"""
    for i in range(N_MELS):
        dut.log_out_i[i].value = int(values[i])
    dut.load_i.value = 1
    await RisingEdge(dut.clk)
    dut.load_i.value = 0

async def collect_frame(dut, timeout=200):
    """Collect all 40 values from CNN output with ready always high"""
    dut.cnn_ready_i.value = 1
    results = []
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.cnn_valid_o.value == 1:
            results.append(dut.cnn_data_o.value.integer)
        if len(results) == N_MELS:
            break
    dut.cnn_ready_i.value = 0
    return results

async def collect_frame_backpressure(dut, ready_pattern, timeout=500):
    """Collect all 40 values with a specific ready pattern"""
    results = []
    cycle   = 0
    for _ in range(timeout):
        dut.cnn_ready_i.value = int(ready_pattern[cycle % len(ready_pattern)])
        await RisingEdge(dut.clk)
        if dut.cnn_valid_o.value == 1 and dut.cnn_ready_i.value == 1:
            results.append(dut.cnn_data_o.value.integer)
        cycle += 1
        if len(results) == N_MELS:
            break
    dut.cnn_ready_i.value = 0
    return results

# ── Tests ─────────────────────────────────────────────────────────

@cocotb.test()
async def test_load_and_drain(dut):
    """Basic: load 40 values, CNN always ready, verify all 40 come out in order"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    # generate known values — use index as value so order is easy to verify
    values = list(range(N_MELS))

    await load_frame(dut, values)
    results = await collect_frame(dut)

    assert len(results) == N_MELS, f"Only got {len(results)}/{N_MELS} values"
    for i in range(N_MELS):
        assert results[i] == values[i], \
            f"mel[{i}]: expected {values[i]} got {results[i]}"
    dut._log.info(f"PASS | load and drain | all {N_MELS} values correct in order")


@cocotb.test()
async def test_frame_sent_signal(dut):
    """frame_sent_o should pulse exactly once after all 40 values are accepted"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    values = [i * 100 for i in range(N_MELS)]
    await load_frame(dut, values)

    frame_sent_count = 0
    dut.cnn_ready_i.value = 1

    for _ in range(200):
        await RisingEdge(dut.clk)
        if dut.frame_sent_o.value == 1:
            frame_sent_count += 1

    assert frame_sent_count == 1, \
        f"frame_sent_o pulsed {frame_sent_count} times, expected exactly 1"
    dut._log.info(f"PASS | frame_sent_o pulsed exactly once")


@cocotb.test()
async def test_cnn_not_ready(dut):
    """CNN holds ready low — buffer should hold valid high and not advance"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    values = list(range(N_MELS))
    await load_frame(dut, values)

    # hold ready low for 10 cycles — valid should stay high, data should stay at index 0
    dut.cnn_ready_i.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
        assert dut.cnn_valid_o.value == 1, "valid dropped while ready was low"
        assert dut.cnn_data_o.value.integer == values[0], \
            f"data changed while ready was low: got {dut.cnn_data_o.value.integer}"

    dut._log.info(f"PASS | buffer holds data when CNN not ready")

    # now drain the rest
    results = await collect_frame(dut)
    assert len(results) == N_MELS, f"Only got {len(results)} after unblocking"
    dut._log.info(f"PASS | drained correctly after unblocking")


@cocotb.test()
async def test_backpressure_alternating(dut):
    """CNN asserts ready every other cycle — all values should still come out correctly"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    rng    = np.random.default_rng(42)
    values = [int(rng.integers(0, 2**OUT_W)) for _ in range(N_MELS)]

    await load_frame(dut, values)

    # alternating ready pattern: 1,0,1,0,...
    ready_pattern = [1, 0]
    results = await collect_frame_backpressure(dut, ready_pattern)

    assert len(results) == N_MELS, f"Only got {len(results)}/{N_MELS} with alternating ready"
    for i in range(N_MELS):
        assert results[i] == values[i], \
            f"mel[{i}]: expected {values[i]} got {results[i]}"
    dut._log.info(f"PASS | alternating backpressure | all values correct")


@cocotb.test()
async def test_two_consecutive_frames(dut):
    """
    Load two frames back to back — simulates real KWS operation where
    log_done fires at the end of each frame and the buffer must reload cleanly.
    Second frame values must not contain anything from the first frame.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(7)

    for frame_idx in range(2):
        values = [int(rng.integers(0, 2**OUT_W)) for _ in range(N_MELS)]

        await load_frame(dut, values)
        results = await collect_frame(dut)

        assert len(results) == N_MELS, \
            f"Frame {frame_idx}: only got {len(results)}/{N_MELS}"
        for i in range(N_MELS):
            assert results[i] == values[i], \
                f"Frame {frame_idx} mel[{i}]: expected {values[i]} got {results[i]}"

        dut._log.info(f"PASS | frame {frame_idx} correct")

    dut._log.info(f"PASS | two consecutive frames both correct")


@cocotb.test()
async def test_load_while_draining(dut):
    """
    New frame arrives (load_i pulses) while previous frame is still being drained.
    In real KWS this shouldn't happen — frame period is much longer than drain time —
    but buffer should handle it gracefully by loading new values immediately.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    rng     = np.random.default_rng(13)
    frame1  = [int(rng.integers(0, 2**OUT_W)) for _ in range(N_MELS)]
    frame2  = [int(rng.integers(0, 2**OUT_W)) for _ in range(N_MELS)]

    # load frame 1 but keep CNN not ready so it doesn't drain
    await load_frame(dut, frame1)
    dut.cnn_ready_i.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # load frame 2 before frame 1 is drained
    await load_frame(dut, frame2)

    # now drain — should get frame 2 values
    results = await collect_frame(dut)

    assert len(results) == N_MELS, f"Only got {len(results)}/{N_MELS}"
    for i in range(N_MELS):
        assert results[i] == frame2[i], \
            f"mel[{i}]: expected frame2 value {frame2[i]} got {results[i]}"

    dut._log.info(f"PASS | overwrite on load_i overwrites buffer correctly")


@cocotb.test()
async def test_valid_low_after_drain(dut):
    """After all 40 values sent, cnn_valid_o should go low and stay low"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    values = list(range(N_MELS))
    await load_frame(dut, values)
    await collect_frame(dut)

    # wait several cycles and check valid stays low
    for _ in range(20):
        await RisingEdge(dut.clk)
        assert dut.cnn_valid_o.value == 0, \
            f"cnn_valid_o went high after buffer was drained"

    dut._log.info(f"PASS | valid stays low after full drain")


@cocotb.test()
async def test_random_backpressure(dut):
    """Random ready signal across 5 frames — stress test the handshake"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    rng   = np.random.default_rng(99)
    fails = 0

    for frame_idx in range(5):
        values = [int(rng.integers(0, 2**OUT_W)) for _ in range(N_MELS)]
        await load_frame(dut, values)

        # random ready pattern for this frame
        ready_pattern = [int(rng.integers(0, 2)) for _ in range(20)]
        # ensure at least half are high so it doesn't take forever
        ready_pattern = [1 if x == 0 and rng.random() > 0.3 else x
                        for x in ready_pattern]

        results = await collect_frame_backpressure(dut, ready_pattern)

        if len(results) != N_MELS:
            fails += 1
            dut._log.info(f"FAIL frame={frame_idx}: only got {len(results)}/{N_MELS}")
            continue

        for i in range(N_MELS):
            if results[i] != values[i]:
                fails += 1
                dut._log.info(
                    f"FAIL frame={frame_idx} mel[{i}]: "
                    f"expected {values[i]} got {results[i]}"
                )

    if fails == 0:
        dut._log.info(f"PASS | 5 frames random backpressure all correct")
    else:
        raise AssertionError(f"{fails} failures across 5 random backpressure frames")