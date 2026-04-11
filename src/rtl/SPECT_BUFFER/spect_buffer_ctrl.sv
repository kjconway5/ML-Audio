module spect_buffer_ctrl #(
    parameter int SPECT_SHIFT = 4,       // first_conv SPECT_SHIFT from export.py

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
    output wire                   cnn_ready_o,   // always high — we are always ready

    // Spectrogram Signals
    // Bank A
    output wire                   sp_a_we,
    output wire [ADDR_W-1:0]      sp_a_waddr,
    output wire signed [7:0]      sp_a_wdata,

    // Bank B
    output wire                   sp_b_we,
    output wire [ADDR_W-1:0]      sp_b_waddr,
    output wire signed [7:0]      sp_b_wdata,

    // FSM Handshake
    output logic                  spect_done,      // full spectrogram written
    output logic                  spect_write_sel  // 0 = writing A / 1 = writing B
);

    localparam int TOTAL_SAMPLES = N_FRAMES * N_MELS;  // 50 × 40 = 2000 (Based on size that Model was trained on

    // Count # of accepted samples
    reg [ADDR_W-1:0] wr_addr;

    // Quantization (based on export.py
    wire signed [IN_W-1:0] shifted = cnn_data_i >>> SPECT_SHIFT;

    wire signed [7:0] quant_val =
        (shifted > $signed(10'd127))  ? 8'sh7F :
        (shifted < $signed(-10'd128)) ? 8'sh80 :
        shifted[7:0];

    assign cnn_ready_o = 1'b1; 

    wire wr_en = cnn_valid_i;  // write whenever upstream sends a valid sample

    assign sp_a_we    = wr_en & ~spect_write_sel;  // writing A when sel=0
    assign sp_b_we    = wr_en &  spect_write_sel;  // writing B when sel=1

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
            spect_done <= 1'b0;  // default: deassert each cycle

            if (cnn_valid_i) begin
                if (wr_addr == ADDR_W'(TOTAL_SAMPLES - 1)) begin
                    // Last sample of this window just written
                    wr_addr         <= '0;
                    spect_done      <= 1'b1;
                    spect_write_sel <= ~spect_write_sel;  // flip for next window
                end else begin
                    wr_addr <= wr_addr + 1'b1;
                end
            end
        end
    end

endmodule
