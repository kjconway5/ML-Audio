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

    // Uncomment for SDF
    // initial $sdf_annotate("../final/sdf/nom_tt_025C_3v30/chip_top.sdf",
    //                       u_chip_top, , "sdf.log", "MAXIMUM");

    // Comment this out when trying to run SDF

    // Power-window VCD dumping — cocotb sets power_dump_en when entering
    // the active workload window, clears it after.
    reg power_dump_en = 0;
    always @(posedge power_dump_en) begin
        $dumpfile("power_window.vcd");
        $dumpvars(0, u_chip_top);              // ← everything in the chip
        $display("[POWER] dumping VCD to power_window.vcd");
    end
    always @(negedge power_dump_en) begin
        $dumpoff;
        $display("[POWER] VCD dump complete");
        $finish;
    end
endmodule