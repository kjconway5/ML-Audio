module mel_coeff_sram #(
    parameter int COEFF_DEPTH = 640, // 40 filters x 16 coeffs
    parameter int INDEX_DEPTH = 80,  // 40 starts + 40 ends
    parameter int COEFF_W     = 16,  // 16-bit coefficients 
    parameter int INDEX_W     = 8,   // 8-bit start/end bin indices
    parameter int COEFF_AW    = 10,  // bits for holding 640 coefficients
    parameter int INDEX_AW    = 7    // bits for holding 80 mel filter start/end values
)(
    input  wire        clk_i,

    // Flash write port for coefficient SRAM
    input  wire        flash_coeff_we_i,
    input  wire [COEFF_AW-1:0] flash_coeff_addr_i,
    input  wire [COEFF_W-1:0]  flash_coeff_data_i,

    // Flash write port for start/end bin 
    input  wire        flash_index_we_i,
    input  wire [INDEX_AW-1:0]  flash_index_addr_i,
    input  wire [INDEX_W-1:0]   flash_index_data_i,

    // Runtime read for coeffs
    input  wire [COEFF_AW-1:0] coeff_addr_i,
    output wire [COEFF_W-1:0]  coeff_data_o,

    // Runtime read for indices
    input  wire [INDEX_AW-1:0]  index_addr_i,
    output wire [INDEX_W-1:0]   index_data_o
);


`ifndef SYNTHESIS

    // Simulation behavioral model loaded from hex files
    reg [COEFF_W-1:0] coeff_mem [0:COEFF_DEPTH-1];
    reg [INDEX_W-1:0]  index_mem  [0:INDEX_DEPTH-1];

    initial begin
        $readmemh("data/mel_coeffs.hex", coeff_mem);
        // mel_indices.hex = mel_starts.hex followed by mel_ends.hex in same file
        $readmemh("data/mel_indices.hex",   index_mem);
    end

    // Registered read to match SRAM latency
    reg [COEFF_W-1:0] coeff_data_r;
    reg [INDEX_W-1:0]  index_data_r;

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
    assign index_data_o  = index_data_r;

`else

    // Coefficient SRAM: 640 coeffs x 16-bits each
    // Two sram1024x8 banks (high byte / low byte)
    // Only 640 of 1024 addresses used
    wire [COEFF_AW-1:0] coeff_addr = flash_coeff_we_i ? flash_coeff_addr_i : coeff_addr_i;
    wire coeff_gwen = flash_coeff_we_i ? 1'b0 : 1'b1;
    wire [7:0] coeff_wen = flash_coeff_we_i ? 8'h00 : 8'hFF;

    wire [7:0] coeff_hi, coeff_lo;

    gf180mcu_ocd_ip_sram__sram1024x8m8wm1 u_coeff_hi (
        .CLK  (clk_i),
        .CEN  (1'b0),
        .GWEN (coeff_gwen),
        .WEN  (coeff_wen),
        .A    (coeff_addr),
        .D    (flash_coeff_data_i[15:8]),
        .Q    (coeff_hi)
    );

    gf180mcu_ocd_ip_sram__sram1024x8m8wm1 u_coeff_lo (
        .CLK  (clk_i),
        .CEN  (1'b0),
        .GWEN (coeff_gwen),
        .WEN  (coeff_wen),
        .A    (coeff_addr),
        .D    (flash_coeff_data_i[7:0]),
        .Q    (coeff_lo)
    );

    assign coeff_data_o = {coeff_hi, coeff_lo};


    // Start/End Bin SRAM: 80 values x 8-bits each
    // Starts in addresses 0-39, Ends in addresses 40-79
    wire [INDEX_AW-1:0] index_addr = flash_index_we_i ? flash_index_addr_i : index_addr_i;
    wire index_gwen = flash_index_we_i ? 1'b0 : 1'b1;
    wire [7:0] index_wen = flash_index_we_i ? 8'h00 : 8'hFF;

    wire [7:0] index_q;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u_index (
        .CLK  (clk_i),
        .CEN  (1'b0),
        .GWEN (index_gwen),
        .WEN  (index_wen),
        .A    (index_addr),
        .D    (flash_index_data_i),
        .Q    (index_q)
    );

    assign index_data_o = index_q;

endmodule