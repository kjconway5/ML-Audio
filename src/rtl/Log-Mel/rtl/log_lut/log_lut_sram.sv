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
    output wire [DATA_W-1:0]   rd_data_o,

    // Test mode
    input  wire [0:0]          test_mode_i,
    input  wire [ADDR_W-1:0]   test_addr_i
);

    wire [ADDR_W-1:0] rd_addr_muxed = test_mode_i ? test_addr_i : rd_addr_i;

    wire [7:0] sram_addr = flash_write_enable_i ? {2'b00, flash_addr_i}
                                                : {2'b00, rd_addr_muxed};
    wire       gwen = flash_write_enable_i ? 1'b0 : 1'b1;
    wire [7:0] wen  = flash_write_enable_i ? 8'h00 : 8'hFF;

`ifdef SIM
    reg [7:0] mem_lo [0:255];
    reg [7:0] mem_hi [0:255];
    reg [7:0] q_lo_r, q_hi_r;

    // Sim-only default load so pipeline tests see usable LUT before any
    // boot-flash. Silicon ignores this block; real runs must flash the LUT
    // over UART (see chip_core.sv boot subsystem).
    reg [15:0] lut_init [0:63];
    integer i;
    initial begin
        $readmemh("log2_lut.hex", lut_init);
        for (i = 0; i < 64; i = i + 1) begin
            mem_lo[i] = lut_init[i][7:0];
            mem_hi[i] = lut_init[i][15:8];
        end
    end

    always @(posedge clk_i) begin
        if (!gwen) begin
            mem_lo[sram_addr] <= flash_write_data_i[7:0];
            mem_hi[sram_addr] <= flash_write_data_i[15:8];
        end else begin
            q_lo_r <= mem_lo[sram_addr];
            q_hi_r <= mem_hi[sram_addr];
        end
    end

    assign rd_data_o = {q_hi_r, q_lo_r};

`else
    wire [7:0] q_lo, q_hi;

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

    // Real GF180 sram256x8 has asynchronous (combinatorial) read output.
    // Register Q here to give 1-cycle read latency matching the SIM model.
    reg [7:0] q_lo_r, q_hi_r;
    always_ff @(posedge clk_i)
        if (gwen) begin  // gwen=1 = read cycle
            q_lo_r <= q_lo;
            q_hi_r <= q_hi;
        end

    assign rd_data_o = {q_hi_r, q_lo_r};

`endif

endmodule
