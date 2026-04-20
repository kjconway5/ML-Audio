module mel_coeff_sram #(
    parameter int COEFF_DEPTH = 640,
    parameter int RANGE_DEPTH = 40,   // one entry per filter
    parameter int COEFF_W     = 16,
    parameter int RANGE_W     = 8,
    parameter int COEFF_AW    = 10,
    parameter int RANGE_AW    = 6     // ceil(log2(40))
)(
    input  wire        clk_i,

    // Flash write for coeff SRAM
    input  wire                  flash_coeff_we_i,
    input  wire [COEFF_AW-1:0]  flash_coeff_addr_i,
    input  wire [COEFF_W-1:0]   flash_coeff_data_i,

    // Flash write for start bin SRAM
    input  wire                  flash_start_we_i,
    input  wire [RANGE_AW-1:0]  flash_start_addr_i,
    input  wire [RANGE_W-1:0]   flash_start_data_i,

    // Flash write for end bin SRAM
    input  wire                  flash_end_we_i,
    input  wire [RANGE_AW-1:0]  flash_end_addr_i,
    input  wire [RANGE_W-1:0]   flash_end_data_i,

    // Runtime read for coeffs
    input  wire [COEFF_AW-1:0]  coeff_addr_i,
    output wire [COEFF_W-1:0]   coeff_data_o,

    // Runtime read for start and end bins (same address, both read simultaneously)
    input  wire [RANGE_AW-1:0]  range_addr_i,
    output wire [RANGE_W-1:0]   start_data_o,
    output wire [RANGE_W-1:0]   end_data_o
);

`ifndef SYNTHESIS

    reg [COEFF_W-1:0] coeff_mem [0:COEFF_DEPTH-1];
    reg [RANGE_W-1:0] start_mem [0:RANGE_DEPTH-1];
    reg [RANGE_W-1:0] end_mem   [0:RANGE_DEPTH-1];

    initial begin
        $readmemh("../../../data/mel_coeffs.hex", coeff_mem);
        $readmemh("../../../data/mel_starts.hex", start_mem);
        $readmemh("../../../data/mel_ends.hex",   end_mem);
    end

    reg [COEFF_W-1:0] coeff_data_r;
    reg [RANGE_W-1:0] start_data_r;
    reg [RANGE_W-1:0] end_data_r;

    always @(posedge clk_i) begin
        if (flash_coeff_we_i)
            coeff_mem[flash_coeff_addr_i] <= flash_coeff_data_i;
        else
            coeff_data_r <= coeff_mem[coeff_addr_i];

        if (flash_start_we_i)
            start_mem[flash_start_addr_i] <= flash_start_data_i;
        else
            start_data_r <= start_mem[range_addr_i];

        if (flash_end_we_i)
            end_mem[flash_end_addr_i] <= flash_end_data_i;
        else
            end_data_r <= end_mem[range_addr_i];
    end

    assign coeff_data_o = coeff_data_r;
    assign start_data_o = start_data_r;
    assign end_data_o   = end_data_r;

`else

    // Coefficient SRAM: 640 x 16-bit, uses two sram1024x8 banks one for hi part and one for low
    wire [COEFF_AW-1:0] coeff_addr = flash_coeff_we_i ? flash_coeff_addr_i : coeff_addr_i;
    wire coeff_gwen = flash_coeff_we_i ? 1'b0 : 1'b1;
    wire [7:0] coeff_wen = flash_coeff_we_i ? 8'h00 : 8'hFF;
    wire [7:0] coeff_hi, coeff_lo;

    gf180mcu_ocd_ip_sram__sram1024x8m8wm1 u_coeff_hi (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (coeff_gwen), .WEN (coeff_wen),
        .A    (coeff_addr), .D  (flash_coeff_data_i[15:8]), .Q (coeff_hi)
    );
    gf180mcu_ocd_ip_sram__sram1024x8m8wm1 u_coeff_lo (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (coeff_gwen), .WEN (coeff_wen),
        .A    (coeff_addr), .D  (flash_coeff_data_i[7:0]),  .Q (coeff_lo)
    );
    assign coeff_data_o = {coeff_hi, coeff_lo};

    // Start bin SRAM: 40 x 8-bit, one sram256x8
    wire [RANGE_AW-1:0] start_addr = flash_start_we_i ? flash_start_addr_i : range_addr_i;
    wire start_gwen = flash_start_we_i ? 1'b0 : 1'b1;
    wire [7:0] start_wen = flash_start_we_i ? 8'h00 : 8'hFF;
    wire [7:0] start_q;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_start (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (start_gwen), .WEN (start_wen),
        .A    (start_addr), .D  (flash_start_data_i), .Q (start_q)
    );
    assign start_data_o = start_q;

    // End bin SRAM: 40 x 8-bit, one sram256x8
    wire [RANGE_AW-1:0] end_addr = flash_end_we_i ? flash_end_addr_i : range_addr_i;
    wire end_gwen = flash_end_we_i ? 1'b0 : 1'b1;
    wire [7:0] end_wen = flash_end_we_i ? 8'h00 : 8'hFF;
    wire [7:0] end_q;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_end (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (end_gwen), .WEN (end_wen),
        .A    (end_addr), .D  (flash_end_data_i), .Q (end_q)
    );
    assign end_data_o = end_q;

`endif
endmodule