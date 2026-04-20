// dscnn_boot_router.sv
//
// Combinational decoder: routes boot_bus_t transactions to the
// correct write port within the DS-CNN module.


module dscnn_boot_router
    import boot_pkg::*;
(
    // Boot bus input
    input  boot_bus_t          boot_i,

    // Weight SRAM boot write port
    output logic               w_boot_we_o,
    output logic [12:0]        w_boot_addr_o,
    output logic [7:0]         w_boot_wdata_o,

    // config write port
    output logic               cfg_boot_we_o,
    output logic [7:0]         cfg_boot_addr_o,
    output logic [7:0]         cfg_boot_wdata_o
);

    always_comb begin
        // Defaults: all write-enables deasserted
        w_boot_we_o     = 1'b0;
        w_boot_addr_o   = 13'h0000;
        w_boot_wdata_o  = 8'h00;

        cfg_boot_we_o    = 1'b0;
        cfg_boot_addr_o  = 8'h00;
        cfg_boot_wdata_o = 8'h00;

        if (boot_i.valid) begin
            case (dscnn_subtarget_e'(boot_i.subtarget))
                DSCNN_WEIGHTS: begin
                    w_boot_we_o    = 1'b1;
                    w_boot_addr_o  = boot_i.addr[12:0];
                    w_boot_wdata_o = boot_i.data[7:0];
                end

                DSCNN_CFG: begin
                    cfg_boot_we_o    = 1'b1;
                    cfg_boot_addr_o  = boot_i.addr[7:0];
                    cfg_boot_wdata_o = boot_i.data[7:0];
                end

                default: ; // unknown subtarget, ignore
            endcase
        end
    end

endmodule