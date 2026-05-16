/*
Original first-window implementation, kept here as a reference.

module spect_buffer_ctrl #(
    parameter int SPECT_SHIFT = 4,

    parameter int N_MELS      = 40,
    parameter int N_FRAMES    = 50,
    parameter int IN_W        = 16,
    parameter int ADDR_W      = 11
)(
    input  wire                clk,
    input  wire                reset,

    input  wire signed [IN_W-1:0] cnn_data_i,
    input  wire                   cnn_valid_i,
    output wire                   cnn_ready_o,

    output wire                   sp_a_we,
    output wire [ADDR_W-1:0]      sp_a_waddr,
    output wire signed [7:0]      sp_a_wdata,

    output wire                   sp_b_we,
    output wire [ADDR_W-1:0]      sp_b_waddr,
    output wire signed [7:0]      sp_b_wdata,

    output logic                  spect_done,
    output logic                  spect_write_sel
);

    localparam int TOTAL_SAMPLES = N_FRAMES * N_MELS;

    reg [ADDR_W-1:0] wr_addr;

    wire signed [IN_W-1:0] shifted = cnn_data_i >>> SPECT_SHIFT;

    wire signed [7:0] quant_val =
        (shifted > $signed(10'd127))  ? 8'sh7F :
        (shifted < $signed(-10'd128)) ? 8'sh80 :
        shifted[7:0];

    assign cnn_ready_o = 1'b1;

    wire wr_en = cnn_valid_i;

    assign sp_a_we    = wr_en & ~spect_write_sel;
    assign sp_b_we    = wr_en &  spect_write_sel;

    assign sp_a_waddr = wr_addr;
    assign sp_b_waddr = wr_addr;
    assign sp_a_wdata = quant_val;
    assign sp_b_wdata = quant_val;

    always_ff @(posedge clk) begin
        if (reset) begin
            wr_addr         <= '0;
            spect_done      <= 1'b0;
            spect_write_sel <= 1'b0;
        end else begin
            spect_done <= 1'b0;

            if (cnn_valid_i) begin
                if (wr_addr == TOTAL_SAMPLES-1) begin
                    wr_addr         <= '0;
                    spect_done      <= 1'b1;
                    spect_write_sel <= ~spect_write_sel;
                end else begin
                    wr_addr <= wr_addr + 1'b1;
                end
            end
        end
    end

endmodule
*/

module spect_buffer_ctrl #(
    parameter int SPECT_SHIFT = 9,       // legacy first_conv SPECT_SHIFT from export.py
    parameter int USE_INPUT_REQUANT = 1,
    parameter int INPUT_QUANT_MULT  = 5817845,
    parameter int INPUT_QUANT_SHIFT = 31,
    parameter int START_FRAME = 37,      // center-ish 50-frame window from a 1-second clip

    parameter int N_MELS      = 40,      // mel bins per frame
    parameter int N_FRAMES    = 50,      // frames per inference window
    parameter int IN_W        = 16,      // width of logmel output
    parameter int ADDR_W      = 11       // must match spectrogram_sram ADDR_W
)(
    input  wire                clk,
    input  wire                reset,

    // Logmel_top signals
    input  wire signed [IN_W-1:0] cnn_data_i,    // one mel bin per cycle
    input  wire                   cnn_valid_i,   // valid handshake
    output wire                   cnn_ready_o,   // always high; no backpressure

    output wire                   sp_a_we,
    output wire [ADDR_W-1:0]      sp_a_waddr,
    output wire signed [7:0]      sp_a_wdata,

    output wire                   sp_b_we,
    output wire [ADDR_W-1:0]      sp_b_waddr,
    output wire signed [7:0]      sp_b_wdata,

    // FSM Handshake
    output logic                  spect_done,      // full selected spectrogram written
    output logic                  spect_write_sel,  // 0 = writing A / 1 = writing B

    // test signals 
    output logic [ADDR_W-1:0]      sp_a_waddr_test,
    output logic signed [7:0]      sp_a_wdata_test,

    output logic [ADDR_W-1:0]      sp_b_waddr_test,
    output logic signed [7:0]      sp_b_wdata_test,
    output logic                   spect_write_sel_test,
    input  logic                   test_mode_audio
);

    localparam int TOTAL_SAMPLES = N_FRAMES * N_MELS;     // 50 x 40 = 2000
    localparam int SKIP_SAMPLES  = START_FRAME * N_MELS;  // 37 x 40 = 1480
    localparam int WINDOW_END    = SKIP_SAMPLES + TOTAL_SAMPLES;
    localparam int COUNT_W       = (WINDOW_END <= 1) ? 1 : $clog2(WINDOW_END + 1);
    localparam logic [COUNT_W-1:0] SKIP_SAMPLES_C = SKIP_SAMPLES;
    localparam logic [COUNT_W-1:0] WINDOW_END_C   = WINDOW_END;
    localparam logic [ADDR_W-1:0]  LAST_WR_ADDR   = TOTAL_SAMPLES - 1;

    logic [ADDR_W-1:0]      wr_addr;
    logic [COUNT_W-1:0]     sample_count;

    // Log-mel values are Q6.10 and nonnegative after bfpexp compensation.
    // The preferred path matches the model QuantStub input scale:
    //   int8 = round((q6_10 / 1024) / input_scale)
    //        = round(q6_10 * INPUT_QUANT_MULT / 2^INPUT_QUANT_SHIFT)
    // The SPECT_SHIFT path is retained as a legacy fallback.
    localparam int PRODUCT_W = IN_W + 32;
    localparam int ROUND_SHIFT = (INPUT_QUANT_SHIFT > 0) ? (INPUT_QUANT_SHIFT - 1) : 0;
    localparam logic [PRODUCT_W-1:0] ONE_PRODUCT_W = {{(PRODUCT_W-1){1'b0}}, 1'b1};
    localparam logic [PRODUCT_W-1:0] INPUT_ROUND_BIAS =
        (INPUT_QUANT_SHIFT > 0) ? (ONE_PRODUCT_W << ROUND_SHIFT) : '0;
    localparam logic [PRODUCT_W-1:0] INT8_MAX = 127;

    wire [IN_W-1:0] cnn_data_u = cnn_data_i[IN_W-1:0];
    wire [31:0] input_quant_mult_u = INPUT_QUANT_MULT;
    wire [PRODUCT_W-1:0] input_product =
        cnn_data_u * input_quant_mult_u;
    wire [PRODUCT_W-1:0] input_requant_u =
        (input_product + INPUT_ROUND_BIAS) >> INPUT_QUANT_SHIFT;

    wire [IN_W-1:0] legacy_shifted_u = cnn_data_u >> SPECT_SHIFT;
    wire [PRODUCT_W-1:0] quant_u =
        USE_INPUT_REQUANT ? input_requant_u :
        {{(PRODUCT_W-IN_W){1'b0}}, legacy_shifted_u};

    wire signed [7:0] quant_val =
        (quant_u > INT8_MAX) ? 8'sh7F :
        $signed({1'b0, quant_u[6:0]});

    wire capture_active = (sample_count >= SKIP_SAMPLES_C) &&
                          (sample_count <  WINDOW_END_C);

    assign cnn_ready_o = 1'b1;

    wire wr_en = cnn_valid_i && capture_active;

    assign sp_a_we    = wr_en & ~spect_write_sel;
    assign sp_b_we    = wr_en &  spect_write_sel;

    assign sp_a_waddr = wr_addr;
    assign sp_b_waddr = wr_addr;
    assign sp_a_wdata = quant_val;
    assign sp_b_wdata = quant_val;

    always_ff @(posedge clk) begin
        if (reset) begin
            wr_addr         <= '0;
            sample_count    <= '0;
            spect_done      <= 1'b0;
            spect_write_sel <= 1'b0;
        end else begin
            spect_done <= 1'b0;

            if (cnn_valid_i) begin
                sample_count <= sample_count + 1'b1;

                if (capture_active) begin
                    if (wr_addr == LAST_WR_ADDR) begin
                        wr_addr         <= '0;
                        sample_count    <= '0;
                        spect_done      <= 1'b1;
                        spect_write_sel <= ~spect_write_sel;
                    end else begin
                        wr_addr <= wr_addr + 1'b1;
                    end
                end
            end
        end
    end

    // test mode to observe at io pins
    always_ff @(posedge clk) begin
        if (reset) begin
            sp_a_waddr_test <= '0;
            sp_a_wdata_test <= '0;
            sp_b_waddr_test <= '0;
            sp_b_wdata_test <= '0;
            spect_write_sel_test <= 1'b0;
        end else if (test_mode_audio) begin
            sp_a_waddr_test <= sp_a_waddr;
            sp_a_wdata_test <= sp_a_wdata;

            sp_b_waddr_test <= sp_b_waddr;
            sp_b_wdata_test <= sp_b_wdata;

            spect_write_sel_test <= spect_write_sel;
        end else begin
            sp_a_waddr_test <= '0;
            sp_a_wdata_test <= '0;
            sp_b_waddr_test <= '0;
            sp_b_wdata_test <= '0;
            spect_write_sel_test <= 1'b0;
        end
    end

endmodule
