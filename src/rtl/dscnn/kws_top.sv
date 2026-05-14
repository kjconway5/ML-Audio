// kws_top.v
// This is the only file Yosys needs as the top for synthesis

module kws_top #(
    parameter DATA_W  = 8,
    parameter ACC_W   = 32,
    parameter ADDR_W  = 14
)(
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

    // test signals 
    input logic test_mode_ml,

    // fsm
    output logic [2:0]fsm_class_test,

    output logic  [ADDR_W-1:0]        fsm_a_waddr_test,
    output logic  signed [DATA_W-1:0] fsm_a_wdata_test,
    output logic  [ADDR_W-1:0]        fsm_a_raddr_test,   
    output logic  signed [DATA_W-1:0] fsm_a_rdata_test,

    output logic  [ADDR_W-1:0]        fsm_b_waddr_test,
    output logic  signed [DATA_W-1:0] fsm_b_wdata_test,
    output logic  [ADDR_W-1:0]        fsm_b_raddr_test,   
    output logic  signed [DATA_W-1:0] fsm_b_rdata_test, 

    output logic  signed [DATA_W-1:0] mac_ifmap_test,
    output logic  signed [DATA_W-1:0] mac_weight_test,
    output logic  signed [ACC_W-1:0]  mac_bias_test,

    output logic  [31:0]              rq_mult_test,
    output logic  [4:0]               rq_shift_test

    output logic [3:0]                state_test,
    output logic [3:0]                layer_test

    // acc 
    output logic signed [ACC_W-1:0]   acc_test

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
    wire [31:0]          rq_mult;
    wire [4:0]           rq_shift;
    wire                 rq_relu_en;
    wire signed [7:0]    rq_out;

    // Debug visibility for cocotb: final classifier GAP accumulators.
    wire signed [31:0]   debug_gap0, debug_gap1, debug_gap2, debug_gap3;
    wire signed [31:0]   debug_gap4, debug_gap5, debug_gap6;

    // Bias signals
    wire [8:0]           bias_addr;   // 9-bit for 32-filter model (up to 295 bias entries)
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

        // test signals
        .test_mode_ml(test_mode_ml),
        .acc_test(acc_test)
    );

    requant inst_rq (
        .acc(mac_acc), .mult(rq_mult), .shift(rq_shift),
        .relu_en(rq_relu_en), .out(rq_out)
    );

    bias_DFFs inst_bias (
        .addr(bias_addr),
        .data(bias_data)
    );

    FSM inst_ctrl (
        .clk(clk), .reset(reset), .start(start),
        .done(done), .class_out(class_out),
        .debug_gap0(debug_gap0), .debug_gap1(debug_gap1),
        .debug_gap2(debug_gap2), .debug_gap3(debug_gap3),
        .debug_gap4(debug_gap4), .debug_gap5(debug_gap5),
        .debug_gap6(debug_gap6),
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
        .rq_mult(rq_mult), .rq_shift(rq_shift), .rq_relu_en(rq_relu_en), .rq_out(rq_out),
        .bias_addr(bias_addr), .bias_data(bias_data), 

        // test signals to be passed up to pins
        .test_mode_ml(test_mode_ml),
        .fsm_class_test(fsm_class_test),
        .fsm_a_waddr_test(fsm_a_waddr_test),
        .fsm_a_wdata_test(fsm_a_wdata_test),
        .fsm_a_raddr_test(fsm_a_raddr_test),   
        .fsm_a_rdata_test(fsm_a_rdata_test),
        .fsm_b_waddr_test(fsm_b_waddr_test),
        .fsm_b_wdata_test(fsm_b_wdata_test),
        .fsm_b_raddr_test(fsm_b_raddr_test),   
        .fsm_b_rdata_test(fsm_b_rdata_test), 
        .mac_ifmap_test(mac_ifmap_test),
        .mac_weight_test(mac_weight_test),
        .mac_bias_test(mac_bias_test),
        .rq_mult_test(rq_mult_test),
        .rq_shift_test(rq_shift_test),
        .state_test(state_test),
        .layer_test(layer_test) 
    );

endmodule
