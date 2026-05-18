// Bias storage for DS-CNN: 295 x 32-bit signed INT32 values.
// Silicon implementation uses four byte-wide 512x8 3.3V SRAM macros.

module bias_SRAM #(
    parameter DEPTH     = 295,
    parameter DATA_W    = 32,
    parameter ADDR_W    = 9,
    parameter BYTE_ADDR_W = ADDR_W + 2,
    parameter BIAS_HEX  = "bias.hex"
)(
    input  wire                         clk,

    // Runtime read port. Data is valid one cycle after addr is presented.
    input  wire [ADDR_W-1:0]            addr,
    output wire signed [DATA_W-1:0]     data,

    // Boot byte-write port. Address is byte-addressed, little-endian.
    input  wire                         we,
    input  wire [BYTE_ADDR_W-1:0]       waddr,
    input  wire [7:0]                   wdata
);

`ifdef SIM

    reg [DATA_W-1:0] mem [0:DEPTH-1];
    reg [ADDR_W-1:0] raddr_q;

    initial begin
        $readmemh(BIAS_HEX, mem);
    end

    always @(posedge clk) begin
        if (we) begin
            case (waddr[1:0])
                2'd0: mem[waddr[BYTE_ADDR_W-1:2]][7:0]   <= wdata;
                2'd1: mem[waddr[BYTE_ADDR_W-1:2]][15:8]  <= wdata;
                2'd2: mem[waddr[BYTE_ADDR_W-1:2]][23:16] <= wdata;
                2'd3: mem[waddr[BYTE_ADDR_W-1:2]][31:24] <= wdata;
            endcase
        end
        raddr_q <= addr;
    end

    assign data = mem[raddr_q];

`else

    localparam NUM_BYTES = 4;

    wire [ADDR_W-1:0] word_addr = we ? waddr[BYTE_ADDR_W-1:2] : addr;
    wire [1:0]        byte_sel  = waddr[1:0];
    wire [7:0]        q_out [0:NUM_BYTES-1];

    genvar gi;
    generate
        for (gi = 0; gi < NUM_BYTES; gi++) begin : gen_bias_bytes
            gf180mcu_ocd_ip_sram__sram512x8m8wm1 inst_bias_sram (
                .CLK  (clk),
                .CEN  (1'b0),
                .GWEN (~(we && (byte_sel == gi[1:0]))),
                .WEN  (8'h00),
                .A    (word_addr),
                .D    (wdata),
                .Q    (q_out[gi])
            );
        end
    endgenerate

    assign data = {q_out[3], q_out[2], q_out[1], q_out[0]};

`endif

endmodule
