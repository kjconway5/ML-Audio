// features_boot_router.sv
//
// Combinational decoder: routes boot_bus_t transactions to the
// correct write port within the features pipeline.


module features_boot_router
    import boot_pkg::*;
(
    // Boot bus input 
    input  boot_bus_t           boot_i,

    // Log LUT boot write port (16-bit)
    output logic                lut_boot_we_o,
    output logic [5:0]          lut_boot_addr_o,     // 64 entries
    output logic [15:0]         lut_boot_wdata_o,

    // Mel coefficient SRAM boot write port (16-bit) 
    output logic                mel_boot_we_o,
    output logic [9:0]          mel_boot_addr_o,     // 640 entries
    output logic [15:0]         mel_boot_wdata_o,

    // Mel start/end bin metadata boot write port (8-bit)
    output logic                meta_boot_we_o,
    output logic [6:0]          meta_boot_addr_o,    // 0..39 = start, 40..79 = end
    output logic [7:0]          meta_boot_wdata_o,

    // Hann window SRAM boot write port (16-bit) [future]
    output logic                hann_boot_we_o,
    output logic [7:0]          hann_boot_addr_o,    // 256 entries
    output logic [15:0]         hann_boot_wdata_o,

    //FFT twiddle SRAM boot write port (16-bit) [future]
    output logic                twid_boot_we_o,
    output logic [15:0]         twid_boot_addr_o,    // sized for all stages combined
    output logic [15:0]         twid_boot_wdata_o
);

    always_comb begin
        // Defaults: all write-enables deasserted
        lut_boot_we_o    = 1'b0;
        lut_boot_addr_o  = 6'h00;
        lut_boot_wdata_o = 16'h0000;

        mel_boot_we_o    = 1'b0;
        mel_boot_addr_o  = 10'h000;
        mel_boot_wdata_o = 16'h0000;

        meta_boot_we_o    = 1'b0;
        meta_boot_addr_o  = 7'h00;
        meta_boot_wdata_o = 8'h00;

        hann_boot_we_o    = 1'b0;
        hann_boot_addr_o  = 8'h00;
        hann_boot_wdata_o = 16'h0000;

        twid_boot_we_o    = 1'b0;
        twid_boot_addr_o  = 16'h0000;
        twid_boot_wdata_o = 16'h0000;

        if (boot_i.valid) begin
            case (features_subtarget_e'(boot_i.subtarget))
                FEAT_LOG_LUT: begin
                    lut_boot_we_o    = 1'b1;
                    lut_boot_addr_o  = boot_i.addr[5:0];
                    lut_boot_wdata_o = boot_i.data;
                end

                FEAT_MEL_COEFF: begin
                    mel_boot_we_o    = 1'b1;
                    mel_boot_addr_o  = boot_i.addr[9:0];
                    mel_boot_wdata_o = boot_i.data;
                end

                FEAT_MEL_META: begin
                    meta_boot_we_o    = 1'b1;
                    meta_boot_addr_o  = boot_i.addr[6:0];
                    meta_boot_wdata_o = boot_i.data[7:0];   // 8-bit target
                end

                FEAT_HANN: begin
                    hann_boot_we_o    = 1'b1;
                    hann_boot_addr_o  = boot_i.addr[7:0];
                    hann_boot_wdata_o = boot_i.data;
                end

                FEAT_TWIDDLES: begin
                    twid_boot_we_o    = 1'b1;
                    twid_boot_addr_o  = boot_i.addr;
                    twid_boot_wdata_o = boot_i.data;
                end

                default: ; // unknown subtarget, ignore
            endcase
        end
    end

endmodule