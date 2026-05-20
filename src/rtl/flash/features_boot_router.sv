// features_boot_router.sv
//
// Combinational decoder: routes boot_bus_t transactions to the
// correct write port within the features pipeline.
//
// Hann window and FFT twiddles are hard ROMs in the current chip
// (stfft.sv / fftstage.v use $readmemh), so they have no flash
// write ports and no subtargets here.


module features_boot_router
    import boot_pkg::*;
(
    input logic                 clk_i,
    input logic                 reset_i,

    // Boot bus input
    input  boot_bus_t           boot_i,

    // Log LUT boot write port (16-bit)
    output logic                lut_boot_we_o,
    output logic [5:0]          lut_boot_addr_o,     // 64 entries
    output logic [15:0]         lut_boot_wdata_o,

    // Mel coefficient SRAM boot write port (16-bit, sparse)
    output logic                mel_boot_we_o,
    output logic [7:0]          mel_boot_addr_o,     // 256 entries
    output logic [15:0]         mel_boot_wdata_o,

    // Mel start/end/offset metadata boot write port (8-bit)
    output logic                meta_boot_we_o,
    output logic [7:0]          meta_boot_addr_o,    // 256 entries
    output logic [7:0]          meta_boot_wdata_o,

    // VAD threshold (32-bit registered output)
    output logic [31:0]         vad_threshold_o,

    // Per-checkpoint input requant multiplier (32-bit registered output).
    // Mirrors the vad_threshold two-write protocol: addr=0 → low16,
    // addr=1 → high16. Reset value = 0 = "fall back to RTL parameter
    // default (5817845)" so a boot that never programs this still works.
    output logic [31:0]         input_quant_mult_o
);

    // VAD threshold register — 32-bit, written as two 16-bit halves
    // because the boot data bus is 16 bits (BOOT_DATA_W). The host
    // sends one FEAT_VAD_THRESH packet whose payload covers both
    // halves; boot_controller's auto-incrementing write_addr_q
    // delivers them as (addr=0, low16) then (addr=1, high16).
    //   threshold == 0x00000000 → VAD disabled (pass-through)
    //   threshold == 0xFFFFFFFF → auto-calibrate (256-frame mean × 2)
    //   any other value         → fixed threshold (frame_energy > thr ⇒ active)
    always_ff @(posedge clk_i) begin
        if (reset_i)
            vad_threshold_o <= 32'd0;
        else if (boot_i.valid && boot_i.subtarget == FEAT_VAD_THRESH) begin
            if (boot_i.addr[0])
                vad_threshold_o[31:16] <= boot_i.data;
            else
                vad_threshold_o[15:0]  <= boot_i.data;
        end
    end

    // Same two-write 16/16 protocol as the VAD threshold above.
    always_ff @(posedge clk_i) begin
        if (reset_i)
            input_quant_mult_o <= 32'd0;
        else if (boot_i.valid && boot_i.subtarget == FEAT_INPUT_QUANT_MULT) begin
            if (boot_i.addr[0])
                input_quant_mult_o[31:16] <= boot_i.data;
            else
                input_quant_mult_o[15:0]  <= boot_i.data;
        end
    end

    always_comb begin
        // Defaults: all write-enables deasserted
        lut_boot_we_o    = 1'b0;
        lut_boot_addr_o  = 6'h00;
        lut_boot_wdata_o = 16'h0000;

        mel_boot_we_o    = 1'b0;
        mel_boot_addr_o  = 8'h00;
        mel_boot_wdata_o = 16'h0000;

        meta_boot_we_o    = 1'b0;
        meta_boot_addr_o  = 8'h00;
        meta_boot_wdata_o = 8'h00;

        if (boot_i.valid) begin
            case (boot_i.subtarget)
                FEAT_LOG_LUT: begin
                    lut_boot_we_o    = 1'b1;
                    lut_boot_addr_o  = boot_i.addr[5:0];
                    lut_boot_wdata_o = boot_i.data;
                end

                FEAT_MEL_COEFF: begin
                    mel_boot_we_o    = 1'b1;
                    mel_boot_addr_o  = boot_i.addr[7:0];
                    mel_boot_wdata_o = boot_i.data;
                end

                FEAT_MEL_META: begin
                    meta_boot_we_o    = 1'b1;
                    meta_boot_addr_o  = boot_i.addr[7:0];
                    meta_boot_wdata_o = boot_i.data[7:0];   // 8-bit target
                end

                default: ; // unknown subtarget, ignore
            endcase
        end
    end

endmodule
