// SPDX-FileCopyrightText: © 2025 XXX Authors
// SPDX-License-Identifier: Apache-2.0

`default_nettype none

module chip_core #(
    parameter int NUM_INPUT_PADS = 12,
    parameter int NUM_BIDIR_PADS = 40,
    parameter int NUM_ANALOG_PADS = 2
    )(
    `ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
    `endif

    input  wire clk,       // clock
    input  wire rst_n,     // reset (active low)

    input  wire [NUM_INPUT_PADS-1:0] input_in,   // Input value
    output wire [NUM_INPUT_PADS-1:0] input_pu,   // Pull-up
    output wire [NUM_INPUT_PADS-1:0] input_pd,   // Pull-down

    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,   // Input value
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,  // Output value
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,   // Output enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,   // Input type (0=CMOS Buffer, 1=Schmitt Trigger)
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,   // Slew rate (0=fast, 1=slow)
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,   // Input enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,   // Pull-up
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,   // Pull-down

    inout  wire [NUM_ANALOG_PADS-1:0] analog  // Analog
);

    import boot_pkg::*;

    // Pad index assignments
    localparam int UART_RX_PAD    = 0;
    localparam int UART_TX_PAD    = 1;
    localparam int KWS_DONE_PAD   = 2;
    localparam int KWS_CLASS0_PAD = 3;
    localparam int KWS_CLASS1_PAD = 4;
    localparam int KWS_CLASS2_PAD = 5;
    localparam int VAD_DROP_PAD   = 6;
    // PDM mic input pads (input_in bus)
    //   input_in[0] = PDM_DATA  : 1-bit bitstream from mic (1 = positive, 0 = negative)
    //   input_in[1] = PDM_VALID : strobe — one pulse per PDM bit
    // TODO: add a bidir pad for CLK output to the mic once we choose DLL/ring oscillator

    // 16 MHz / (115200 * 8) = 17  (8 cycles/bit in sim for speed)
`ifdef SIM
    localparam logic [15:0] UART_PRESCALE = 16'd1;
`else
    localparam logic [15:0] UART_PRESCALE = 16'd17;
`endif

    assign input_pu = '0;
    assign input_pd = '0;

    // Default bidir pad config
    logic [NUM_BIDIR_PADS-1:0] bidir_oe_r;
    logic [NUM_BIDIR_PADS-1:0] bidir_ie_r;
    logic [NUM_BIDIR_PADS-1:0] bidir_out_r;

    assign bidir_oe = bidir_oe_r;
    assign bidir_ie = bidir_ie_r;
    assign bidir_out = bidir_out_r;
    assign bidir_cs = '0;
    assign bidir_sl = '0;
    assign bidir_pu = '0;
    assign bidir_pd = '0;

    wire reset;
    assign reset = ~rst_n;

    // UART pad wiring
    wire uart_rxd = bidir_in[UART_RX_PAD];
    wire uart_txd;

    // KWS outputs
    wire            kws_done;
    wire [2:0]      kws_class_out;

    wire pipeline_vad_frame_drop;

    always_comb begin
        bidir_oe_r  = '1;
        bidir_ie_r  = '0;
        bidir_out_r = '0;
        // UART RX pad: input
        bidir_oe_r[UART_RX_PAD]  = 1'b0;
        bidir_ie_r[UART_RX_PAD]  = 1'b1;
        // UART TX pad: output
        bidir_oe_r[UART_TX_PAD]  = 1'b1;
        bidir_ie_r[UART_TX_PAD]  = 1'b0;
        bidir_out_r[UART_TX_PAD] = uart_txd;
        // KWS output pads - prevents optimizer from removing the design
        bidir_out_r[KWS_DONE_PAD]   = kws_done;
        bidir_out_r[KWS_CLASS0_PAD] = kws_class_out[0];
        bidir_out_r[KWS_CLASS1_PAD] = kws_class_out[1];
        bidir_out_r[KWS_CLASS2_PAD] = kws_class_out[2];
        // VAD DFT - High when a frame is dropped
        bidir_out_r[VAD_DROP_PAD]   = pipeline_vad_frame_drop;
    end

    // Tie off unused inputs so lint stays clean
    logic _unused;
    assign _unused = &{bidir_in, input_in[NUM_INPUT_PADS-1:3], 1'b0};

    // ---- Boot subsystem ----

    wire [7:0] rx_byte;
    wire       rx_valid;
    wire       rx_ready;
    wire       rx_busy;
    wire       rx_overrun;
    wire       rx_frame_err;

    uart_rx u_uart_rx (
        .clk           (clk),
        .rst           (reset),
        .m_axis_tdata  (rx_byte),
        .m_axis_tvalid (rx_valid),
        .m_axis_tready (rx_ready),
        .rxd           (uart_rxd),
        .busy          (rx_busy),
        .overrun_error (rx_overrun),
        .frame_error   (rx_frame_err),
        .prescale      (UART_PRESCALE)
    );

    wire [7:0] boot_tx_byte;
    wire       boot_tx_valid;
    wire       tx_ready;
    wire       tx_busy;

    // Score TX sequencer drives the UART after each KWS inference.
    // When active it takes priority over boot_controller TX.
    logic        score_tx_active;
    logic [7:0]  score_tx_byte;

    wire [7:0] tx_byte  = score_tx_active ? score_tx_byte  : boot_tx_byte;
    wire       tx_valid = score_tx_active ? 1'b1            : boot_tx_valid;

    uart_tx u_uart_tx (
        .clk           (clk),
        .rst           (reset),
        .s_axis_tdata  (tx_byte),
        .s_axis_tvalid (tx_valid),
        .s_axis_tready (tx_ready),
        .txd           (uart_txd),
        .busy          (tx_busy),
        .prescale      (UART_PRESCALE)
    );

    boot_bus_t features_boot;
    boot_bus_t dscnn_boot;
    wire       boot_done;

    boot_controller u_boot_ctrl (
        .clk_i           (clk),
        .reset_i         (reset),
        .rx_byte_i       (rx_byte),
        .rx_valid_i      (rx_valid),
        .rx_ready_o      (rx_ready),
        .tx_byte_o       (boot_tx_byte),
        .tx_valid_o      (boot_tx_valid),
        .tx_ready_i      (tx_ready),
        .features_boot_o (features_boot),
        .dscnn_boot_o    (dscnn_boot),
        .boot_done_o     (boot_done),
        .pkt_valid_o     (),
        .last_target_o   (),
        .last_addr_o     (),
        .last_len_o      ()
    );

    // Features pipeline flash write ports
    wire        flash_log_lut_we;
    wire [5:0]  flash_log_lut_addr;
    wire [15:0] flash_log_lut_data;

    wire        flash_mel_coeff_we;
    wire [7:0]  flash_mel_coeff_addr;
    wire [15:0] flash_mel_coeff_data;

    wire        flash_mel_index_we;
    wire [7:0]  flash_mel_index_addr;
    wire [7:0]  flash_mel_index_data;

    wire [31:0] vad_threshold;


    features_boot_router u_feat_router (
        .clk_i             (clk),
        .reset_i           (reset),
        .boot_i            (features_boot),
        .lut_boot_we_o     (flash_log_lut_we),
        .lut_boot_addr_o   (flash_log_lut_addr),
        .lut_boot_wdata_o  (flash_log_lut_data),
        .mel_boot_we_o     (flash_mel_coeff_we),
        .mel_boot_addr_o   (flash_mel_coeff_addr),
        .mel_boot_wdata_o  (flash_mel_coeff_data),
        .meta_boot_we_o    (flash_mel_index_we),
        .meta_boot_addr_o  (flash_mel_index_addr),
        .meta_boot_wdata_o (flash_mel_index_data),
        .vad_threshold_o   (vad_threshold)
    );

    // DS-CNN flash write ports
    wire        flash_weight_we;
    wire [12:0] flash_weight_addr;
    wire [7:0]  flash_weight_data;
    wire        flash_bias_we;
    wire [10:0] flash_bias_addr;
    wire [7:0]  flash_bias_data;

    wire        flash_cfg_we;
    wire [7:0]  flash_cfg_addr;
    wire [7:0]  flash_cfg_data;

    dscnn_boot_router u_dscnn_router (
        .boot_i           (dscnn_boot),
        .w_boot_we_o      (flash_weight_we),
        .w_boot_addr_o    (flash_weight_addr),
        .w_boot_wdata_o   (flash_weight_data),
        .b_boot_we_o      (flash_bias_we),
        .b_boot_addr_o    (flash_bias_addr),
        .b_boot_wdata_o   (flash_bias_data),
        .cfg_boot_we_o    (flash_cfg_we),
        .cfg_boot_addr_o  (flash_cfg_addr),
        .cfg_boot_wdata_o (flash_cfg_data)
    );

    // Hold the feature pipeline in reset until boot completes so it cannot
    // produce spectrograms from uninitialized LogMel memories.
    // Keep KWS out of boot_done-gated reset so UART boot writes can program
    // its weights/config before inference starts.
    wire inference_reset = reset | ~boot_done;

    // ---- PDM microphone wiring ----
    // Sign-extend 1-bit PDM data to ±full-scale 16-bit for the CIC.
    // The test (and real mic) drive input_in[1] as a strobe (one pulse per PDM bit)
    // and input_in[0] as the data bit — identical to how test_full_pipeline_top.py
    // drives valid_i / data_i directly on the sub-module.
    wire [15:0] pdm_word  = input_in[0] ? 16'h7FFF : 16'h8000;
    wire        pdm_valid = input_in[1];

    wire            spect_done;

    // Auto-start KWS inference one cycle after spect_done fires.
    // spect_ready inside the FSM is registered (set 1 cycle after spect_done),
    // so we delay start by one cycle to guarantee both flags are true together.
    logic kws_start;
    always_ff @(posedge clk) begin
        if (inference_reset)
            kws_start <= 1'b0;
        else
            kws_start <= spect_done;
    end

    // ---- Pipeline + KWS ----

    // Spectrogram signals
    wire            sp_a_we;
    wire [10:0]     sp_a_waddr;
    wire signed [7:0] sp_a_wdata;

    wire            sp_b_we;
    wire [10:0]     sp_b_waddr;
    wire signed [7:0] sp_b_wdata;

    wire            spect_write_sel;

    // Dangling outputs for full_pipeline_top
    wire pipeline_ready;
    wire [15:0] mel_compensated;
    wire mel_compensated_valid;
    wire pipeline_vad_active;


    full_pipeline_top #(
        .IW_STFFT   (16),
        .OW_STFFT   (16),
        .FFT_SIZE   (256),
        .N_MELS     (40),
        .N_BINS     (129),
        .N_FRAMES   (50),
        .START_FRAME(37),
        .SPECT_SHIFT(9),
        .USE_INPUT_REQUANT(1),
        .INPUT_QUANT_MULT (5805163),
        .INPUT_QUANT_SHIFT(31),
        .ADDR_W     (11),
        .LUT_FRAC   (6)
    ) pipeline_inst (
        .clk_i              (clk),
        .reset_i            (inference_reset),
        .data_i             (pdm_word),
        .valid_i            (pdm_valid),
        .ready_o            (pipeline_ready),
        .sp_a_we            (sp_a_we),
        .sp_a_waddr         (sp_a_waddr),
        .sp_a_wdata         (sp_a_wdata),
        .sp_b_we            (sp_b_we),
        .sp_b_waddr         (sp_b_waddr),
        .sp_b_wdata         (sp_b_wdata),
        .spect_done         (spect_done),
        .spect_write_sel    (spect_write_sel),
        .mel_compensated_o       (mel_compensated),
        .mel_compensated_valid_o (mel_compensated_valid),
        // VAD ports
        .vad_threshold_i    (vad_threshold), 
        .vad_active_o       (pipeline_vad_active), 
        // Flash ports
        .flash_mel_coeff_we_i   (flash_mel_coeff_we),
        .flash_mel_coeff_addr_i (flash_mel_coeff_addr),
        .flash_mel_coeff_data_i (flash_mel_coeff_data),
        .flash_mel_index_we_i   (flash_mel_index_we),
        .flash_mel_index_addr_i (flash_mel_index_addr),
        .flash_mel_index_data_i (flash_mel_index_data),
        .flash_log_lut_we_i     (flash_log_lut_we),
        .flash_log_lut_addr_i   (flash_log_lut_addr),
        .flash_log_lut_data_i   (flash_log_lut_data),
        // VAD DFT Ports
        .dft_vad_obs_en_i   (input_in[2]),
        .vad_frame_drop_ol  (pipeline_vad_frame_drop),
        // TODO: wire test_mode_audio to a real pad/scan chain when DFT is complete
        .test_mode_audio    (1'b0)
    );

    // GAP score wires from kws_top (registered, valid when kws_done is high)
    wire signed [31:0] gap_score_0, gap_score_1, gap_score_2, gap_score_3;
    wire signed [31:0] gap_score_4, gap_score_5, gap_score_6;

    kws_top kws_inst (
        .clk(clk),
        .reset(reset),
        .start(kws_start),
        .done(kws_done),
        .class_out(kws_class_out),
        // Layer config write port
        .cfg_we(flash_cfg_we),
        .cfg_addr(flash_cfg_addr),
        .cfg_wdata(flash_cfg_data),
        // Weight SRAM write port
        .w_we(flash_weight_we),
        .w_waddr(flash_weight_addr),
        .w_wdata(flash_weight_data),
        .b_we(flash_bias_we),
        .b_waddr(flash_bias_addr),
        .b_wdata(flash_bias_data),
        .spect_done(spect_done),
        .spect_write_sel(spect_write_sel),
        .sp_a_we(sp_a_we),
        .sp_a_waddr(sp_a_waddr),
        .sp_a_wdata(sp_a_wdata),
        .sp_b_we(sp_b_we),
        .sp_b_waddr(sp_b_waddr),
        .sp_b_wdata(sp_b_wdata),
        // Enable test capture so debug_gap*_test tracks live accumulators
        .test_mode_ml(1'b1),
        .debug_gap0_test(gap_score_0),
        .debug_gap1_test(gap_score_1),
        .debug_gap2_test(gap_score_2),
        .debug_gap3_test(gap_score_3),
        .debug_gap4_test(gap_score_4),
        .debug_gap5_test(gap_score_5),
        .debug_gap6_test(gap_score_6)
    );

    // ---- Score TX sequencer ------------------------------------------------
    // On the rising edge of kws_done, latch all 7 GAP scores and stream them
    // out via UART TX as a 29-byte packet:
    //   byte  0     : 0xDA  (header marker)
    //   bytes 1-4   : gap_score_0 (class 0 = "no"),      big-endian INT32
    //   bytes 5-8   : gap_score_1 (class 1 = "off")
    //   bytes 9-12  : gap_score_2 (class 2 = "on")
    //   bytes 13-16 : gap_score_3 (class 3 = "silence")
    //   bytes 17-20 : gap_score_4 (class 4 = "unknown")
    //   bytes 21-24 : gap_score_5 (class 5 = "wow")
    //   bytes 25-28 : gap_score_6 (class 6 = "yes")

    logic [7:0]  score_buf [0:28];
    logic [4:0]  score_idx;   // 0..28
    logic        kws_done_r;

    always_ff @(posedge clk) begin
        if (reset) begin
            score_tx_active <= 1'b0;
            score_tx_byte   <= 8'h00;
            score_idx       <= 5'd0;
            kws_done_r      <= 1'b0;
        end else begin
            kws_done_r <= kws_done;

            if (!score_tx_active && kws_done && !kws_done_r) begin
                // Rising edge: latch packet. debug_gap*_test were updated at
                // the previous posedge, so pre-clock values here are final.
                score_buf[0]  <= 8'hDA;
                score_buf[1]  <= gap_score_0[31:24]; score_buf[2]  <= gap_score_0[23:16];
                score_buf[3]  <= gap_score_0[15:8];  score_buf[4]  <= gap_score_0[7:0];
                score_buf[5]  <= gap_score_1[31:24]; score_buf[6]  <= gap_score_1[23:16];
                score_buf[7]  <= gap_score_1[15:8];  score_buf[8]  <= gap_score_1[7:0];
                score_buf[9]  <= gap_score_2[31:24]; score_buf[10] <= gap_score_2[23:16];
                score_buf[11] <= gap_score_2[15:8];  score_buf[12] <= gap_score_2[7:0];
                score_buf[13] <= gap_score_3[31:24]; score_buf[14] <= gap_score_3[23:16];
                score_buf[15] <= gap_score_3[15:8];  score_buf[16] <= gap_score_3[7:0];
                score_buf[17] <= gap_score_4[31:24]; score_buf[18] <= gap_score_4[23:16];
                score_buf[19] <= gap_score_4[15:8];  score_buf[20] <= gap_score_4[7:0];
                score_buf[21] <= gap_score_5[31:24]; score_buf[22] <= gap_score_5[23:16];
                score_buf[23] <= gap_score_5[15:8];  score_buf[24] <= gap_score_5[7:0];
                score_buf[25] <= gap_score_6[31:24]; score_buf[26] <= gap_score_6[23:16];
                score_buf[27] <= gap_score_6[15:8];  score_buf[28] <= gap_score_6[7:0];
                score_idx       <= 5'd0;
                score_tx_active <= 1'b1;
                score_tx_byte   <= 8'hDA;
            end else if (score_tx_active) begin
                score_tx_byte <= score_buf[score_idx];
                if (tx_ready) begin
                    if (score_idx == 5'd28)
                        score_tx_active <= 1'b0;
                    else
                        score_idx <= score_idx + 1'b1;
                end
            end
        end
    end

endmodule
`default_nettype wire
