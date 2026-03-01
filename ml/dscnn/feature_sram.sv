// feature_sram.sv
// Ping-pong dual buffer for intermediate feature maps between layers
// Both banks are read and written at runtime — hard SRAM macro required
//
// TODO: Instantiate GF180 SRAM macros here
//
// Required interface:
//   - Depth per bank:  12,000 entries  (24ch × 25 × 20, 14-bit address)
//   - Width:           8-bit data      (INT8)
//   - Two independent banks (A and B) for ping-pong buffering
//   - Each bank needs simultaneous read and write capability
//     → requires either a true dual-port macro or
//       a single-port macro with careful timing (no same-cycle R/W conflict)
//
// Ping-pong scheme:
//   buf_sel=0 → controller reads from A, writes to B
//   buf_sel=1 → controller reads from B, writes to A
//   buf_sel flips in NEXT_LAYER state of layer_controller.sv
//
// Feature map address layout within each bank:
//   addr = ch * ofmap_H * ofmap_W + oh * ofmap_W + ow
//   Max address = 24 * 25 * 20 - 1 = 11,999
//
// NOTE: Input feature map (mel spectrogram) is written into the active
// buffer before inference begins, then first_conv reads from it.
// The input has shape (1ch × 49 × 40 = 1,960 values) which fits
// comfortably within the 12,000 entry bank.

module feature_sram #(
    parameter DEPTH  = 12000,
    parameter DATA_W = 8,
    parameter ADDR_W = 14
)(
    input  wire              clk,

    // Bank A
    input  wire              a_we,
    input  wire [ADDR_W-1:0] a_waddr,
    input  wire [DATA_W-1:0] a_wdata,
    input  wire [ADDR_W-1:0] a_raddr,
    output wire [DATA_W-1:0] a_rdata,

    // Bank B
    input  wire              b_we,
    input  wire [ADDR_W-1:0] b_waddr,
    input  wire [DATA_W-1:0] b_wdata,
    input  wire [ADDR_W-1:0] b_raddr,
    output wire [DATA_W-1:0] b_rdata
);

    // TODO: Replace with GF180 SRAM macro instantiation
    //
    // gf180mcu_fd_ip_sram__sram512x8m8wm1 u_bank_a (
    //     .CLK  (clk),
    //     .CEN  (1'b0),
    //     .WEN  (~a_we),
    //     .A    (a_addr),     // mux between a_waddr and a_raddr based on a_we
    //     .D    (a_wdata),
    //     .Q    (a_rdata)
    // );
    //
    // NOTE: 12,000 entries requires cascaded macros.
    // If using single-port macros, the controller must ensure
    // read and write never occur on the same bank in the same cycle.
    // The current FSM design guarantees this since COMPUTE only reads
    // and WRITE_OFMAP only writes — they are separate FSM states.

    assign a_rdata = 8'h00; // placeholder
    assign b_rdata = 8'h00; // placeholder

endmodule