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
    output wire [INDEX_W-1:0]   index_data_o,

    // Test mode
    input  wire                 test_mode_i,
    input  wire [COEFF_AW-1:0]  test_coeff_addr_i,
    input  wire [INDEX_AW-1:0]  test_index_addr_i
);

`ifdef SIM

    reg [COEFF_W-1:0]  coeff_mem [0:COEFF_DEPTH-1];
    reg [INDEX_W-1:0]  index_mem [0:INDEX_DEPTH-1];

    initial begin
        //$readmemh("../../../data/mel_coeffs_sparse.hex", coeff_mem);
        //$readmemh("../../../data/mel_indices.hex", index_mem);
        $readmemh("mel_coeffs_sparse.hex", coeff_mem);
        $readmemh("mel_indices.hex", index_mem);
    end

    wire [COEFF_AW-1:0] coeff_rd_addr = test_mode_i ? test_coeff_addr_i : coeff_addr_i;
    wire [INDEX_AW-1:0] index_rd_addr = test_mode_i ? test_index_addr_i : index_addr_i;

    reg [COEFF_W-1:0] coeff_data_r;
    reg [INDEX_W-1:0] index_data_r;

    always @(posedge clk_i) begin
        if (flash_coeff_we_i)
            coeff_mem[flash_coeff_addr_i] <= flash_coeff_data_i;
        else
            coeff_data_r <= coeff_mem[coeff_rd_addr];

        if (flash_index_we_i)
            index_mem[flash_index_addr_i] <= flash_index_data_i;
        else
            index_data_r <= index_mem[index_rd_addr];
    end

    assign coeff_data_o = coeff_data_r;
    assign index_data_o = index_data_r;

`else

    wire [COEFF_AW-1:0] coeff_rd_muxed = test_mode_i ? test_coeff_addr_i : coeff_addr_i;
    wire [INDEX_AW-1:0] index_rd_muxed = test_mode_i ? test_index_addr_i : index_addr_i;

    // Coefficient SRAM: 246 entries x 16-bit, two sram256x8 banks (hi/lo byte)
    wire [COEFF_AW-1:0] coeff_addr = flash_coeff_we_i ? flash_coeff_addr_i : coeff_rd_muxed;
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

    // Real GF180 sram256x8 has asynchronous (combinatorial) read output.
    // Register Q here to give 1-cycle read latency matching the SIM model.
    reg [7:0] coeff_hi_r, coeff_lo_r;
    always_ff @(posedge clk_i)
        if (coeff_gwen) begin  // coeff_gwen=1 = read cycle
            coeff_hi_r <= coeff_hi;
            coeff_lo_r <= coeff_lo;
        end

    assign coeff_data_o = {coeff_hi_r, coeff_lo_r};

    // Index SRAM: 256 x 8-bit, one sram256x8
    // Layout: [0:39] = start_bin, [40:79] = end_bin, [80:119] = coeff_offset
    wire [INDEX_AW-1:0] index_addr = flash_index_we_i ? flash_index_addr_i : index_rd_muxed;
    wire index_gwen = flash_index_we_i ? 1'b0 : 1'b1;
    wire [7:0] index_wen = flash_index_we_i ? 8'h00 : 8'hFF;
    wire [7:0] index_q;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_index (
        .CLK  (clk_i), .CEN  (1'b0),
        .GWEN (index_gwen), .WEN (index_wen),
        .A    (index_addr), .D  (flash_index_data_i), .Q (index_q)
    );

    reg [7:0] index_q_r;
    always_ff @(posedge clk_i)
        if (index_gwen)  // index_gwen=1 = read cycle
            index_q_r <= index_q;

    assign index_data_o = index_q_r;

`endif
endmodule