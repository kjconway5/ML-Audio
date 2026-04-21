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
    output logic [7:0]          meta_boot_wdata_o
);

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
