// Read-only weight storage: 4,296 × 8-bit INT8 values
// Implemented as 5× cascaded gf180mcu_ocd_ip_sram__sram1024x8m8wm1 macros
// (5 × 1024 = 5120) 

module weight_sram #(
    parameter DEPTH  = 4296,
    parameter DATA_W = 8,
    parameter ADDR_W = 13   // covers 0–8191; valid range 0–4295
)(
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output wire [DATA_W-1:0] data
);

    localparam NUM_BANKS = 5;

    // Upper bits [12:10] select which 1024-entry bank 
    // Lower bits  [9:0] offset within each entry bank 
    wire [2:0]  bank_sel  = addr[12:10];  
    wire [9:0]  bank_addr = addr[9:0];    

    // Per-bank chip enables (active-low CEN)
    wire [NUM_BANKS-1:0] cen;  // one per macro
    genvar gi;
    generate
        for (gi = 0; gi < NUM_BANKS; gi++) begin : gen_cen
            assign cen[gi] = (bank_sel == gi[2:0]) ? 1'b0 : 1'b1;      // Enable one bank for reading 
        end
    endgenerate

    wire [7:0] q_out [NUM_BANKS-1:0];     // Array of output buses 

    generate
        for (gi = 0; gi < NUM_BANKS; gi++) begin : gen_weight_banks
            gf180mcu_ocd_ip_sram__sram1024x8m8wm1 inst_weight_sram (
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
    reg [2:0] bank_sel_q;   
    always_ff @(posedge clk)       
        bank_sel_q <= bank_sel;

    assign data = q_out[bank_sel_q];

endmodule