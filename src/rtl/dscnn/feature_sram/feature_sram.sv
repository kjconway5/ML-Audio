// Ping-pong  feature-map buffer: 2 banks × 12,000 × 8-bit INT8
// Each bank implemented as 12× cascaded sram1024x8 macros
// (12 × 1024 = 12,288 capacity; valid range 0–11,999)


// Read latency: 1 cycle 
// Write latency: 1 cycle 

module feature_sram #(
    parameter DEPTH  = 12000,
    parameter DATA_W = 8,
    parameter ADDR_W = 14   // covers 0–16383; valid range 0–11,999
)(
    input  wire              clk,

    // Bank A ports
    input  wire              a_we,
    input  wire [ADDR_W-1:0] a_waddr,
    input  wire [DATA_W-1:0] a_wdata,
    input  wire [ADDR_W-1:0] a_raddr,
    output wire [DATA_W-1:0] a_rdata,

    // Bank B ports
    input  wire              b_we,
    input  wire [ADDR_W-1:0] b_waddr,
    input  wire [DATA_W-1:0] b_wdata,
    input  wire [ADDR_W-1:0] b_raddr,
    output wire [DATA_W-1:0] b_rdata
);

`ifdef SIM

    // Behavioral model for simulation — simple reg array, 1-cycle read latency
    reg [DATA_W-1:0] mem_a [0:DEPTH-1];
    reg [DATA_W-1:0] mem_b [0:DEPTH-1];

    reg [ADDR_W-1:0] a_raddr_q;
    reg [ADDR_W-1:0] b_raddr_q;

    always @(posedge clk) begin
        if (a_we)
            mem_a[a_waddr] <= a_wdata;
        a_raddr_q <= a_raddr;

        if (b_we)
            mem_b[b_waddr] <= b_wdata;
        b_raddr_q <= b_raddr;
    end

    assign a_rdata = mem_a[a_raddr_q];
    assign b_rdata = mem_b[b_raddr_q];

`else

    // 12 x 1024 = 12,288
    localparam NUM_BANKS = 12;

    wire [ADDR_W-1:0] a_addr = a_we ? a_waddr : a_raddr;
    wire [ADDR_W-1:0] b_addr = b_we ? b_waddr : b_raddr;

    // Upper bits select macro instance; lower 10 bits are bank-specific offset
    wire [3:0] a_bank_sel  = a_addr[13:10];
    wire [9:0] a_bank_addr = a_addr[9:0];
    wire [3:0] b_bank_sel  = b_addr[13:10];
    wire [9:0] b_bank_addr = b_addr[9:0];


    wire [NUM_BANKS-1:0] a_cen;   // chip enable (active-low) (which bank to access)
    wire [NUM_BANKS-1:0] a_gwen;  // global write enable (active-low = write) (read/write)
    wire [NUM_BANKS-1:0] b_cen;
    wire [NUM_BANKS-1:0] b_gwen;

    wire [7:0] a_q [NUM_BANKS-1:0];
    wire [7:0] b_q [NUM_BANKS-1:0];

    genvar gi;
    generate
        for (gi = 0; gi < NUM_BANKS; gi++) begin : gen_feat_banks

            // Bank A
            assign a_cen[gi]  = (a_bank_sel == gi[3:0]) ? 1'b0 : 1'b1;
            assign a_gwen[gi] = a_we ? 1'b0 : 1'b1;

            // 1024x8 3.3V SRAM Macro
            gf180mcu_ocd_ip_sram__sram1024x8m8wm1 inst_feature_sram (
                .CLK  (clk),
                .CEN  (a_cen[gi]),
                .GWEN (a_gwen[gi]),
                .WEN  (8'h00),      // write all 8 bits when GWEN=0
                .A    (a_bank_addr),
                .D    (a_wdata),
                .Q    (a_q[gi])
                `ifdef USE_POWER_PINS
                ,.VDD  (1'b1),
                ,.VSS  (1'b0)
                `endif
            );

            // Bank B
            assign b_cen[gi]  = (b_bank_sel == gi[3:0]) ? 1'b0 : 1'b1;
            assign b_gwen[gi] = b_we ? 1'b0 : 1'b1;

            gf180mcu_ocd_ip_sram__sram1024x8m8wm1 inst_feature_sram2 (
                .CLK  (clk),
                .CEN  (b_cen[gi]),
                .GWEN (b_gwen[gi]),
                .WEN  (8'h00),
                .A    (b_bank_addr),
                .D    (b_wdata),
                .Q    (b_q[gi])
                `ifdef USE_POWER_PINS
                ,.VDD  (1'b1),
                ,.VSS  (1'b0)
                `endif
            );

        end
    endgenerate

    // account for 1 cycle read latency
    reg [3:0] a_bank_sel_q;
    reg [3:0] b_bank_sel_q;

    always_ff @(posedge clk) begin
        a_bank_sel_q <= a_bank_sel;
        b_bank_sel_q <= b_bank_sel;
    end

    assign a_rdata = a_q[a_bank_sel_q];
    assign b_rdata = b_q[b_bank_sel_q];

`endif

endmodule