// rtl/fft_data_ram.sv
// Stage ping-pong dual-port wrapper for GF180 single-port SRAMs
// Each logical port = 2 physical banks, alternating read/write by FFT stage
// No simultaneous read+write to any physical bank

module fft_data_ram (
    input  wire        clk,
    input  wire        rst,
    // Stage control — toggle every FFT stage (connect to R2FFT SB_NEXT_STAGE)
    input  wire        next_stage,
    // Read port
    input  wire        ract,
    input  wire [6:0]  ra,
    output reg  [31:0] rdata,
    // Write port
    input  wire        wact,
    input  wire [6:0]  wa,
    input  wire [31:0] wdata
);

`ifdef SYNTHESIS
    // ASIC implementation: two physical SRAM banks per logical port
    reg stage_sel;  // 0: read from A, write to B
                    // 1: read from B, write to A

    always @(posedge clk) begin
        if (rst)
            stage_sel <= 1'b0;
        else if (next_stage)
            stage_sel <= ~stage_sel;
    end

    // Bank A control
    wire cen_a  = ~((ract && stage_sel == 1'b1) ||
                    (wact && stage_sel == 1'b0));
    wire gwen_a = ~(wact && stage_sel == 1'b0);
    wire [6:0] addr_a = (wact && stage_sel == 1'b0) ? wa : ra;

    // Bank B control
    wire cen_b  = ~((ract && stage_sel == 1'b0) ||
                    (wact && stage_sel == 1'b1));
    wire gwen_b = ~(wact && stage_sel == 1'b1);
    wire [6:0] addr_b = (wact && stage_sel == 1'b1) ? wa : ra;

    wire [7:0] wen = 8'h00;

    // Bank A — 4× sram128x8 for 32-bit width
    wire [7:0] qa0, qa1, qa2, qa3;

    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_a_b0 (
        .CLK(clk), .CEN(cen_a), .GWEN(gwen_a), .WEN(wen),
        .A(addr_a), .D(wdata[7:0]),   .Q(qa0));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_a_b1 (
        .CLK(clk), .CEN(cen_a), .GWEN(gwen_a), .WEN(wen),
        .A(addr_a), .D(wdata[15:8]),  .Q(qa1));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_a_b2 (
        .CLK(clk), .CEN(cen_a), .GWEN(gwen_a), .WEN(wen),
        .A(addr_a), .D(wdata[23:16]), .Q(qa2));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_a_b3 (
        .CLK(clk), .CEN(cen_a), .GWEN(gwen_a), .WEN(wen),
        .A(addr_a), .D(wdata[31:24]), .Q(qa3));

    // Bank B — 4× sram128x8 for 32-bit width
    wire [7:0] qb0, qb1, qb2, qb3;

    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b_b0 (
        .CLK(clk), .CEN(cen_b), .GWEN(gwen_b), .WEN(wen),
        .A(addr_b), .D(wdata[7:0]),   .Q(qb0));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b_b1 (
        .CLK(clk), .CEN(cen_b), .GWEN(gwen_b), .WEN(wen),
        .A(addr_b), .D(wdata[15:8]),  .Q(qb1));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b_b2 (
        .CLK(clk), .CEN(cen_b), .GWEN(gwen_b), .WEN(wen),
        .A(addr_b), .D(wdata[23:16]), .Q(qb2));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b_b3 (
        .CLK(clk), .CEN(cen_b), .GWEN(gwen_b), .WEN(wen),
        .A(addr_b), .D(wdata[31:24]), .Q(qb3));

    // Read mux — registered bank select for 1-cycle SRAM latency
    reg stage_sel_q;
    always @(posedge clk)
        stage_sel_q <= stage_sel;

    always @(posedge clk) begin
        if (ract) begin
            if (stage_sel_q == 1'b0)
                rdata <= {qb3, qb2, qb1, qb0};  // read from B when sel=0
            else
                rdata <= {qa3, qa2, qa1, qa0};  // read from A when sel=1
        end
    end

`else
    // ----------------------------------------------------------------
    // Simulation: behavioral true dual-port
    // ----------------------------------------------------------------
    reg [31:0] mem [0:127];

    integer i;
    initial begin
        for (i = 0; i < 128; i = i + 1)
            mem[i] = 32'h0;
        rdata = 32'h0;
    end

    always @(posedge clk) begin
        if (wact) mem[wa] <= wdata;
        if (ract) rdata   <= mem[ra];
    end

`endif

endmodule