// Read-only weight storage: 4,296 × 8-bit INT8 values
// Implemented as 9× cascaded gf180mcu_fd_ip_sram__sram512x8m8wm1 macros
// (9 × 512 = 4608 capacity; only addresses 0–4295 are valid)


// Weight layout:
//   0x0000–0x03BF : first_conv   (960  entries)
//   0x03C0–0x0497 : dw block 0   (216  entries)
//   0x0498–0x06B7 : pw block 0   (576  entries)
//   0x06B8–0x078F : dw block 1   (216  entries)
//   0x0790–0x09AF : pw block 1   (576  entries)
//   0x09B0–0x0A87 : dw block 2   (216  entries)
//   0x0A88–0x0CA7 : pw block 2   (576  entries)
//   0x0CA8–0x0D7F : dw block 3   (216  entries)
//   0x0D80–0x0F9F : pw block 3   (576  entries)
//   0x0FA0–0x1047 : classifier   (168  entries)

module weight_sram #(
    parameter DEPTH  = 4296,
    parameter DATA_W = 8,
    parameter ADDR_W = 13   // covers 0–8191; valid range 0–4295
)(
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output wire [DATA_W-1:0] data
);

    // Upper bits [12:9] select which 512-entry bank (0–8)
    // Lower bits  [8:0] offset within each entry bank 
    wire [3:0] bank_sel  = addr[12:9];  
    wire [8:0] bank_addr = addr[8:0];

    // Per-bank chip enables (active-low CEN)
    wire [8:0] cen;  // one per macro
    genvar gi;
    generate
        for (gi = 0; gi < 9; gi++) begin : gen_cen
            assign cen[gi] = (bank_sel == gi[3:0]) ? 1'b0 : 1'b1;      // Enable one bank for reading 
        end
    endgenerate

    wire [7:0] q_out [8:0];     // Array of output buses 

    generate
        for (gi = 0; gi < 9; gi++) begin : gen_weight_banks
            gf180mcu_fd_ip_sram__sram512x8m8wm1 u_weight (
                .CLK  (clk),
                .CEN  (cen[gi]),
                .GWEN (1'b1),       // always read
                .WEN  (8'hFF),      // all bits write-disabled
                .A    (bank_addr),
                .D    (8'h00),      // unused
                .Q    (q_out[gi]),
                .VDD  (1'b1),       // Check with GF180 PDK on what is required for RTL during P&R 
                .VSS  (1'b0)        
            );
        end
    endgenerate

     // Account for 1 cycle read latency 
    reg [3:0] bank_sel_q;
    always_ff @(posedge clk)       
        bank_sel_q <= bank_sel;

    assign data = q_out[bank_sel_q];

endmodule