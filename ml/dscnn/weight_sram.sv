// weight_sram.sv
// Stores all 4,296 INT8 weight values for all 10 layers
// Weights are read-only constants — never written at runtime
//
// TODO: Instantiate GF180 SRAM macro here
// Options:
//   - Case statement → synthesizes to DFFs (functional but larger area)
//   - Hard SRAM macro (gf180mcu_fd_ip_sram) → lower area, preferred
//
// Required interface:
//   - Depth:  4,296 entries  (13-bit address)
//   - Width:  8-bit data     (INT8)
//   - Ports:  clk, addr[12:0], data_out[7:0]
//   - Read latency: 1 cycle
//
// Weight layout in memory (from export.py):
//   0x0000 – 0x03BF  : first_conv         (960  values, shape 24×1×10×4)
//   0x03C0 – 0x0497  : ds_blocks.0.dw     (216  values, shape 24×1×3×3)
//   0x0498 – 0x06B7  : ds_blocks.0.pw     (576  values, shape 24×24×1×1)
//   0x06B8 – 0x078F  : ds_blocks.1.dw     (216  values, shape 24×1×3×3)
//   0x0790 – 0x09AF  : ds_blocks.1.pw     (576  values, shape 24×24×1×1)
//   0x09B0 – 0x0A87  : ds_blocks.2.dw     (216  values, shape 24×1×3×3)
//   0x0A88 – 0x0CA7  : ds_blocks.2.pw     (576  values, shape 24×24×1×1)
//   0x0CA8 – 0x0D7F  : ds_blocks.3.dw     (216  values, shape 24×1×3×3)
//   0x0D80 – 0x0F9F  : ds_blocks.3.pw     (576  values, shape 24×24×1×1)
//   0x0FA0 – 0x1047  : classifier          (168  values, shape 7×24×1×1)

module weight_sram #(
    parameter DEPTH  = 4296,
    parameter DATA_W = 8,
    parameter ADDR_W = 13
)(
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output wire [DATA_W-1:0] data
);

    // TODO: Replace with GF180 SRAM macro instantiation
    // gf180mcu_fd_ip_sram__sram512x8m8wm1 u_weight_sram (
    //     .CLK  (clk),
    //     .CEN  (1'b0),
    //     .WEN  (1'b1),    // always read
    //     .A    (addr),
    //     .D    (8'b0),    // unused, write never occurs
    //     .Q    (data)
    // );
    //
    // NOTE: For depths > 512 you will need to cascade multiple macro
    // instances and use the upper address bits to select between them.
    // 4296 entries requires at least 9 × 512×8 macros cascaded,
    // or fewer instances of a larger macro if available in the PDK.

    assign data = 8'h00; // placeholder until macro is instantiated

endmodule