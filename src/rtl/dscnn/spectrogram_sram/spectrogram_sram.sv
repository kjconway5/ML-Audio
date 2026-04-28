// Ping-pong spectrogram buffer: 2 banks × 2000 × 8-bit INT8
// Each bank implemented as 2× cascaded sram1024x8 macros
// (2 × 1024 = 2048 capacity; valid range 0–1,999)

// Ownership model:
//   - Preprocessor owns the WRITE bank (spect_write_sel)
//   - layer_controller reads from the OPPOSITE bank
//   - Preprocessor pulses spect_done when a full spectrogram is written;
//     layer_controller latches it as its 'start' trigger and knows which
//     bank to read from (= ~spect_write_sel at time of spect_done)


// Read latency : 1 cycle
// Write latency: 1 cycle

module spectrogram_sram #(
    parameter DEPTH  = 2000,
    parameter DATA_W = 8,
    parameter ADDR_W = 11   // covers 0–2047; valid range 0–1,999
)(
    input  wire              clk,

    // bank A ports
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

    localparam NUM_BANKS = 2;

    // Mux address: write takes priority (only one is valid per cycle)
    wire [ADDR_W-1:0] a_addr = a_we ? a_waddr : a_raddr;
    wire [ADDR_W-1:0] b_addr = b_we ? b_waddr : b_raddr;

    // Upper bit selects macro instance; lower 10 bits are bank-specific offset
    wire        a_bank_sel  = a_addr[10];
    wire [9:0]  a_bank_addr = a_addr[9:0];
    wire        b_bank_sel  = b_addr[10];
    wire [9:0]  b_bank_addr = b_addr[9:0];

    wire [NUM_BANKS-1:0] a_cen;
    wire [NUM_BANKS-1:0] a_gwen;
    wire [NUM_BANKS-1:0] b_cen;
    wire [NUM_BANKS-1:0] b_gwen;

    wire [7:0] a_q [NUM_BANKS-1:0];
    wire [7:0] b_q [NUM_BANKS-1:0];

    genvar gi;
    generate
        for (gi = 0; gi < NUM_BANKS; gi++) begin : gen_spect_banks

            // Bank A
            assign a_cen[gi]  = (a_bank_sel == gi[0]) ? 1'b0 : 1'b1;
            assign a_gwen[gi] = a_we ? 1'b0 : 1'b1;

            gf180mcu_ocd_ip_sram__sram1024x8m8wm1 inst_spectrogram_sram (
                .CLK  (clk),
                .CEN  (a_cen[gi]),
                .GWEN (a_gwen[gi]),
                .WEN  (8'h00),
                .A    (a_bank_addr),
                .D    (a_wdata),
                .Q    (a_q[gi])
                //`ifdef USE_POWER_PINS
                //, .VDD  (1'b1)
                //, .VSS  (1'b0)
                //`endif
            );

            // Bank B
            assign b_cen[gi]  = (b_bank_sel == gi[0]) ? 1'b0 : 1'b1;
            assign b_gwen[gi] = b_we ? 1'b0 : 1'b1;

            gf180mcu_ocd_ip_sram__sram1024x8m8wm1 inst_spectrogram_sram2 (
                .CLK  (clk),
                .CEN  (b_cen[gi]),
                .GWEN (b_gwen[gi]),
                .WEN  (8'h00),
                .A    (b_bank_addr),
                .D    (b_wdata),
                .Q    (b_q[gi])
                //`ifdef USE_POWER_PINS
                //, .VDD  (1'b1)
                //, .VSS  (1'b0)
                //`endif
            );

        end
    endgenerate

    // 1-cycle read-latency register for bank-select mux
    reg a_bank_sel_q;
    reg b_bank_sel_q;

    always_ff @(posedge clk) begin
        a_bank_sel_q <= a_bank_sel;
        b_bank_sel_q <= b_bank_sel;
    end

    assign a_rdata = a_q[a_bank_sel_q];
    assign b_rdata = b_q[b_bank_sel_q];

`endif

endmodule