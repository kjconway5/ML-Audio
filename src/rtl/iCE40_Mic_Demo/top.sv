`timescale 1ns / 1ps
`default_nettype none

module top #(
    parameter IW = 16,
    parameter CIC_WIDTH = 16,
    parameter CIC_N = 5,
    parameter CIC_M = 1,
    parameter CIC_R = 63,
    parameter FIR_TAPS = 32,
    parameter OW = 32
)(
    input  wire                 clk,
    input  wire                 rst,

    // PDM input (already converted to signed samples externally)
    input  wire signed [IW-1:0] pdm_sample,
    input  wire                 pdm_valid,

    // Output audio (16 kHz domain)
    output wire signed [OW-1:0] audio_out,
    output wire                 audio_valid
);


    localparam CIC_REG_WIDTH = CIC_WIDTH + $clog2((CIC_R*CIC_M)**CIC_N);

    wire signed [CIC_REG_WIDTH-1:0] cic_out_tdata;
    wire cic_out_tvalid;
    wire cic_in_tready;

    cic_decimator #(
        .WIDTH(CIC_WIDTH),
        .RMAX(CIC_R),
        .M(CIC_M),
        .N(CIC_N)
    ) cic_inst (
        .clk(clk),
        .rst(rst),

        .input_tdata(pdm_sample),
        .input_tvalid(pdm_valid),
        .input_tready(cic_in_tready),

        .output_tdata(cic_out_tdata),
        .output_tvalid(cic_out_tvalid),
        .output_tready(1'b1),

        .rate(CIC_R)
    );

    // CIC grows a lot, must scale down can change if needed
    localparam SHIFT = CIC_N * $clog2(CIC_R);

    wire signed [IW-1:0] fir_in_sample;

    assign fir_in_sample = cic_out_tdata >>> SHIFT;


    wire signed [OW-1:0] fir_out;

    fastfir #(
        .NTAPS(FIR_TAPS),
        .IW(IW),
        .TW(IW),
        .OW(OW),
        .FIXED_TAPS(1)
    ) fir_inst (
        .i_clk(clk),
        .i_reset(rst),

        .i_tap_wr(1'b0),
        .i_tap(0),

        .i_ce(cic_out_tvalid),
        .i_sample(fir_in_sample),

        .o_result(fir_out)
    );


    assign audio_out   = fir_out;
    assign audio_valid = cic_out_tvalid;

endmodule

`default_nettype wire