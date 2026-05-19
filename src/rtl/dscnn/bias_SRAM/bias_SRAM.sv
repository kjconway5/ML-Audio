// Bias storage for DS-CNN.
// 16-filter models fit in two sram512x8 macros. Each read returns one
// 16-bit halfword, so the 32-bit bias word is assembled over two cycles.
// update_rtl.py sets DEPTH and ADDR_W from the exported model's bias count.

module bias_SRAM #(
    parameter DEPTH     = 151,
    parameter DATA_W    = 32,
    parameter ADDR_W    = 8,
    parameter BYTE_ADDR_W = ADDR_W + 2,
    parameter BIAS_HEX  = "bias.hex"
)(
    input  wire                         clk,

    // Runtime read port. Data is valid two cycles after addr is presented.
    input  wire [ADDR_W-1:0]            addr,
    input  wire                         read_high,
    output wire signed [DATA_W-1:0]     data,

    // Boot byte-write port. Address is byte-addressed, little-endian.
    input  wire                         we,
    input  wire [BYTE_ADDR_W-1:0]       waddr,
    input  wire [7:0]                   wdata
);

`ifdef SIM

    reg [DATA_W-1:0] mem [0:DEPTH-1];
    reg [ADDR_W-1:0] raddr_q0, raddr_q1;

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
        raddr_q0 <= addr;
        raddr_q1 <= raddr_q0;
    end

    assign data = mem[raddr_q1];

`else

    localparam NUM_BYTES = 2;
    localparam HALF_ADDR_W = 9;

`ifndef SYNTHESIS
    initial begin
        if (DATA_W != 32)
            $fatal(1, "bias_SRAM compact 2x512 mode requires DATA_W=32");
        if (DEPTH > 256)
            $fatal(1, "bias_SRAM compact 2x512 mode supports at most 256 32-bit biases");
        if (ADDR_W > 8)
            $fatal(1, "bias_SRAM compact 2x512 mode expects ADDR_W<=8");
    end
`endif

    reg [15:0] low_half_q;

    wire [ADDR_W-1:0] word_addr = we ? waddr[BYTE_ADDR_W-1:2] : addr;
    wire [7:0]        word_addr_8 = word_addr[7:0];
    wire              half_sel  = we ? waddr[1] : read_high;
    wire              byte_sel  = waddr[0];
    wire [HALF_ADDR_W-1:0] half_addr = {word_addr_8, half_sel};
    wire [7:0]        q_out [0:NUM_BYTES-1];

    always @(posedge clk) begin
        if (read_high && !we)
            low_half_q <= {q_out[1], q_out[0]};
    end

    genvar gi;
    generate
        for (gi = 0; gi < NUM_BYTES; gi++) begin : gen_bias_bytes
            gf180mcu_ocd_ip_sram__sram512x8m8wm1 inst_bias_sram (
                .CLK  (clk),
                .CEN  (1'b0),
                .GWEN (~(we && (byte_sel == gi[0]))),
                .WEN  (8'h00),
                .A    (half_addr),
                .D    (wdata),
                .Q    (q_out[gi])
            );
        end
    endgenerate

    assign data = {q_out[1], q_out[0], low_half_q};

`endif

endmodule
