`timescale 1ns/1ps
module chip_top_sdf_wrapper (
    inout         VDD,
    inout         VSS,
    input  [11:0] input_PAD,
    inout  [39:0] bidir_PAD,
    input         clk_PAD,
    input         rst_n_PAD
);
    chip_top u_chip_top (
        .VDD       (VDD),
        .VSS       (VSS),
        .input_PAD (input_PAD),
        .bidir_PAD (bidir_PAD),
        .clk_PAD   (clk_PAD),
        .rst_n_PAD (rst_n_PAD)
    );

`ifdef SDF_ENABLED
    `include "sdf_annotate.v"
`endif

    // Debug waveform dumping, armed via +kws_fst_path=<file> (KWS_WAVES=1 in
    // the make environment swaps vvp to -fst). +kws_dump_start_ns=<t> delays
    // the dump start so multi-hour runs only capture the window of interest.
    // Mutually exclusive with +power_vcd_path (one $dumpfile per sim).
    string kws_fst_path;
    longint kws_dump_start_ns;
    initial begin
        if ($value$plusargs("kws_fst_path=%s", kws_fst_path)) begin
            if (!$value$plusargs("kws_dump_start_ns=%d", kws_dump_start_ns))
                kws_dump_start_ns = 0;
            if (kws_dump_start_ns > 0)
                #(kws_dump_start_ns);
            $dumpfile(kws_fst_path);
            $dumpvars(0, u_chip_top);
            $display("[KWS_WAVES] dump to %0s armed at t=%0t", kws_fst_path, $time);
        end
    end

    // Power-window VCD dumping. Register the dump hierarchy at time 0,
    // then capture only the cocotb-controlled power window.
    reg power_dump_en = 0;
    reg power_vcd_enabled = 0;
    string power_vcd_path;

    initial begin
        if ($value$plusargs("power_vcd_path=%s", power_vcd_path)) begin
            power_vcd_enabled = 1;
            $dumpfile(power_vcd_path);
            $dumpvars(0, chip_top_sdf_wrapper);
            #1;
            $dumpoff;
            $display("[POWER] VCD armed at %0s", power_vcd_path);
        end
    end

    always @(posedge power_dump_en) begin
        if (power_vcd_enabled) begin
            $dumpon;
            $display("[POWER] dumping VCD to %0s", power_vcd_path);
        end
    end

    always @(negedge power_dump_en) begin
        if (power_vcd_enabled) begin
            $dumpoff;
            $dumpflush;
            $display("[POWER] VCD dump complete");
        end
    end
endmodule
