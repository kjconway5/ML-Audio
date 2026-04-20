module mel_coeff_sram #(
    parameter int COEFF_DEPTH = 256,   // was 640, now sparse-packed (246 used)
    parameter int INDEX_DEPTH = 256,   // single SRAM: starts[0:39], ends[40:79], offsets[80:119]
    parameter int COEFF_W     = 16,
    parameter int INDEX_W     = 8,
    parameter int COEFF_AW    = 8,     // ceil(log2(256))
    parameter int INDEX_AW    = 8      // ceil(log2(256))
)(
    input  wire        clk_i,

    // Flash write for coeff SRAM
    input  wire                  flash_coeff_we_i,
    input  wire [COEFF_AW-1:0]  flash_coeff_addr_i,
    input  wire [COEFF_W-1:0]   flash_coeff_data_i,

    // Flash write for index SRAM (starts, ends, and offsets all in one SRAM)
    input  wire                  flash_index_we_i,
    input  wire [INDEX_AW-1:0]  flash_index_addr_i,
    input  wire [INDEX_W-1:0]   flash_index_data_i,

    // Runtime read for coeffs
    input  wire [COEFF_AW-1:0]  coeff_addr_i,
    output wire [COEFF_W-1:0]   coeff_data_o,

    // Runtime read for index (start, end, or offset depending on address)
    input  wire [INDEX_AW-1:0]  index_addr_i,
    output wire [INDEX_W-1:0]   index_data_o
);

`ifndef SYNTHESIS

    reg [COEFF_W-1:0]  coeff_mem [0:COEFF_DEPTH-1];
    reg [INDEX_W-1:0]  index_mem [0:INDEX_DEPTH-1];

    initial begin
        $readmemh("../../../data/mel_coeffs_sparse.hex", coeff_mem);
        $readmemh("../../../data/mel_indices.hex", index_mem);
    end

    reg [COEFF_W-1:0] coeff_data_r;
    reg [INDEX_W-1:0] index_data_r;

    always @(posedge clk_i) begin
        if (flash_coeff_we_i)
            coeff_mem[flash_coeff_addr_i] <= flash_coeff_data_i;
        else
            coeff_data_r <= coeff_mem[coeff_addr_i];

        if (flash_index_we_i)
            index_mem[flash_index_addr_i] <= flash_index_data_i;
        else
            index_data_r <= index_mem[index_addr_i];
    end

    assign coeff_data_o = coeff_data_r;
    assign index_data_o = index_data_r;

`else

    // Coefficient SRAM: 246 entries x 16-bit, two sram256x8 banks (hi/lo byte)
    wire [COEFF_AW-1:0] coeff_addr = flash_coeff_we_i ? flash_coeff_addr_i : coeff_addr_i;
    wire coeff_gwen = flash_coeff_we_i ? 1'b0 : 1'b1;
    wire [7:0] coeff_wen = flash_coeff_we_i ? 8'h00 : 8'hFF;
    wire [7:0] coeff_hi, coeff_lo;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_coeff_hi (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (coeff_gwen), .WEN (coeff_wen),
        .A    (coeff_addr), .D  (flash_coeff_data_i[15:8]), .Q (coeff_hi)
    );
    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_coeff_lo (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (coeff_gwen), .WEN (coeff_wen),
        .A    (coeff_addr), .D  (flash_coeff_data_i[7:0]),  .Q (coeff_lo)
    );
    assign coeff_data_o = {coeff_hi, coeff_lo};

    // Index SRAM: 256 x 8-bit, one sram256x8
    // Layout: [0:39] = start_bin, [40:79] = end_bin, [80:119] = coeff_offset
    wire [INDEX_AW-1:0] index_addr = flash_index_we_i ? flash_index_addr_i : index_addr_i;
    wire index_gwen = flash_index_we_i ? 1'b0 : 1'b1;
    wire [7:0] index_wen = flash_index_we_i ? 8'h00 : 8'hFF;
    wire [7:0] index_q;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_index (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (index_gwen), .WEN (index_wen),
        .A    (index_addr), .D  (flash_index_data_i), .Q (index_q)
    );
    assign index_data_o = index_q;

`endif
endmodule