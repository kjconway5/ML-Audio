module log_lut_sram #(
    parameter int LUT_DEPTH = 64,
    parameter int DATA_W    = 16,
    parameter int ADDR_W    = 6
)(
    input  wire [0:0] clk_i,

    // Flash write
    input  wire [0:0]          flash_write_enable_i,
    input  wire [ADDR_W-1:0]   flash_addr_i,
    input  wire [DATA_W-1:0]   flash_write_data_i,

    // Runtime read
    input  wire [ADDR_W-1:0]   rd_addr_i,
    output wire [DATA_W-1:0]   rd_data_o
);

    wire [7:0] sram_addr = flash_write_enable_i ? {2'b00, flash_addr_i}
                                                : {2'b00, rd_addr_i};
    wire       gwen = flash_write_enable_i ? 1'b0 : 1'b1;
    wire [7:0] wen  = flash_write_enable_i ? 8'h00 : 8'hFF;

    wire [7:0] q_lo, q_hi;

`ifndef SYNTHESIS
    reg [7:0] mem_lo [0:255];
    reg [7:0] mem_hi [0:255];
    reg [7:0] q_lo_r, q_hi_r;

    always @(posedge clk_i) begin
        if (!gwen) begin
            mem_lo[sram_addr] <= flash_write_data_i[7:0];
            mem_hi[sram_addr] <= flash_write_data_i[15:8];
        end else begin
            q_lo_r <= mem_lo[sram_addr];
            q_hi_r <= mem_hi[sram_addr];
        end
    end

    assign q_lo = q_lo_r;
    assign q_hi = q_hi_r;
`else
    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_lut_lo (
        .CLK  (clk_i),
        .CEN  (1'b0),
        .GWEN (gwen),
        .WEN  (wen),
        .A    (sram_addr),
        .D    (flash_write_data_i[7:0]),
        .Q    (q_lo)
    );

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_lut_hi (
        .CLK  (clk_i),
        .CEN  (1'b0),
        .GWEN (gwen),
        .WEN  (wen),
        .A    (sram_addr),
        .D    (flash_write_data_i[15:8]),
        .Q    (q_hi)
    );
`endif

    assign rd_data_o = {q_hi, q_lo};

endmodule