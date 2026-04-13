// kws_top.v
// This is the only file Yosys needs as the top for synthesis

module kws_top (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        done,
    output wire [2:0]  class_out,

    // SERV config signals 
    input  wire        cfg_we,
    input  wire [7:0]  cfg_addr,
    input  wire [7:0]  cfg_wdata,

    // Weight SRAM write port (connects to UART)
    input  wire        w_we,
    input  wire [12:0] w_waddr,
    input  wire [7:0]  w_wdata,

    // spect_buffer_ctrl → FSM
    input  wire        spect_done,
    input  wire        spect_write_sel,

    // Spectrogram SRAM write ports 
    // Bank A 
    input  wire        sp_a_we,
    input  wire [10:0] sp_a_waddr,
    input  wire signed [7:0] sp_a_wdata,

    // Bank B 
    input  wire        sp_b_we,
    input  wire [10:0] sp_b_waddr,
    input  wire signed [7:0] sp_b_wdata
    );

    // weight SRAM signals
    wire [12:0]          w_raddr;         
    wire signed [7:0]    w_rdata;         

    // Spectrogram SRAM signals 
    wire [10:0]          ss_a_raddr;
    wire signed [7:0]    ss_a_rdata;

    wire [10:0]          ss_b_raddr;
    wire signed [7:0]    ss_b_rdata;

    // Feature Map SRAM signals 
    wire                 fs_a_we;
    wire [13:0]          fs_a_waddr, fs_a_raddr;
    wire signed [7:0]    fs_a_wdata, fs_a_rdata;

    wire                 fs_b_we;
    wire [13:0]          fs_b_waddr, fs_b_raddr;
    wire signed [7:0]    fs_b_wdata, fs_b_rdata;

    // MAC signals (scalar: one INT8 × INT8 per cycle)
    wire                 mac_en, mac_clear;
    wire signed [7:0]    mac_ifmap, mac_weight;
    wire signed [31:0]   mac_bias, mac_acc;

    // Requant Signals
    wire [4:0]           rq_shift;
    wire                 rq_relu_en;
    wire signed [7:0]    rq_out;

    // Bias signals
    wire [7:0]           bias_addr;
    wire signed [31:0]   bias_data;

    // Shared spectrogram read address 
    assign ss_b_raddr = ss_a_raddr;

    spectrogram_sram inst_specram (
        .clk(clk),
        .a_we(sp_a_we), .a_waddr(sp_a_waddr), .a_wdata(sp_a_wdata),
        .a_raddr(ss_a_raddr), .a_rdata(ss_a_rdata),
        .b_we(sp_b_we), .b_waddr(sp_b_waddr), .b_wdata(sp_b_wdata),
        .b_raddr(ss_b_raddr), .b_rdata(ss_b_rdata)
    );

    weight_sram inst_wsram (
        .clk  (clk),
        .we   (w_we),
        .waddr(w_waddr),
        .wdata(w_wdata),
        .raddr(w_raddr),
        .rdata(w_rdata)
    );

    feature_sram inst_fsram (
        .clk(clk),
        .a_we(fs_a_we), .a_waddr(fs_a_waddr), .a_wdata(fs_a_wdata),
        .a_raddr(fs_a_raddr), .a_rdata(fs_a_rdata),
        .b_we(fs_b_we), .b_waddr(fs_b_waddr), .b_wdata(fs_b_wdata),
        .b_raddr(fs_b_raddr), .b_rdata(fs_b_rdata)
    );

    mac_array inst_mac (
        .clk(clk), .reset(reset),
        .en(mac_en), .clear(mac_clear),
        .ifmap(mac_ifmap), .weight(mac_weight),
        .bias(mac_bias), .acc(mac_acc)
    );

    requant inst_rq (
        .acc(mac_acc), .shift(rq_shift),
        .relu_en(rq_relu_en), .out(rq_out)
    );

    bias_DFFs inst_bias (
        .addr(bias_addr),
        .data(bias_data)
    );

    FSM inst_ctrl (
        .clk(clk), .reset(reset), .start(start),
        .done(done), .class_out(class_out),
        .cfg_we(cfg_we), .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata),
        .spect_done(spect_done), .spect_write_sel(spect_write_sel),
        .weights_ready(1'b1),
        .sp_raddr(ss_a_raddr),
        .sp_a_rdata(ss_a_rdata), .sp_b_rdata(ss_b_rdata),
        .w_addr(w_raddr), .w_data(w_rdata),
        .fs_a_we(fs_a_we), .fs_a_waddr(fs_a_waddr), .fs_a_wdata(fs_a_wdata),
        .fs_a_raddr(fs_a_raddr), .fs_a_rdata(fs_a_rdata),
        .fs_b_we(fs_b_we), .fs_b_waddr(fs_b_waddr), .fs_b_wdata(fs_b_wdata),
        .fs_b_raddr(fs_b_raddr), .fs_b_rdata(fs_b_rdata),
        .mac_en(mac_en), .mac_clear(mac_clear),
        .mac_ifmap(mac_ifmap), .mac_weight(mac_weight),
        .mac_bias(mac_bias), .mac_acc(mac_acc),
        .rq_shift(rq_shift), .rq_relu_en(rq_relu_en), .rq_out(rq_out),
        .bias_addr(bias_addr), .bias_data(bias_data)
    );

endmodule
