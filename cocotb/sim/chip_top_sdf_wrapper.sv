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
