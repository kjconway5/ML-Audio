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
    localparam int DFT_DEBUG_W = 31; // bidir[39:9]

    // Pad index assignments
    localparam int UART_RX_PAD    = 0;
    localparam int UART_TX_PAD    = 1;
    localparam int KWS_DONE_PAD   = 2;
    localparam int KWS_CLASS0_PAD = 3;
    localparam int KWS_CLASS1_PAD = 4;
    localparam int KWS_CLASS2_PAD = 5;
    localparam int VAD_DROP_PAD   = 6;

    localparam int AUDIO_TEST_MODE_PAD  = 7; 
    localparam int ML_TEST_MODE_PAD     = 8;


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


    localparam int ADDR_W = 11;
    localparam int LUT_FRAC = 6;
    localparam int N_MELS = 40;
    localparam int CIC_REG_W = 16;
    localparam int FIR_OW    = 16;
    localparam int POWER_W   = 31;
    localparam int DATA_W    = 8;
    localparam int ACC_W     = 32;

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

    wire test_mode_audio;
    assign test_mode_audio = bidir_in[AUDIO_TEST_MODE_PAD];

    wire test_mode_ml;
    assign test_mode_ml = bidir_in[ML_TEST_MODE_PAD];

    wire [5:0] dft_sel = input_in[8:3];
    logic [DFT_DEBUG_W-1:0] dft_debug_bus;

    always_comb begin
        bidir_oe_r  = '0;
        bidir_ie_r  = '0;
        bidir_out_r = '0;

        // input pads
        bidir_ie_r[UART_RX_PAD] = 1'b1;
        bidir_ie_r[AUDIO_TEST_MODE_PAD] = 1'b1;
        bidir_ie_r[ML_TEST_MODE_PAD] = 1'b1;

        // UART TX output
        bidir_oe_r[UART_TX_PAD]  = 1'b1;
        bidir_out_r[UART_TX_PAD] = uart_txd;

        if (test_mode_audio || test_mode_ml) begin
            bidir_oe_r[39:9]  = '1;
            bidir_out_r[39:9] = dft_debug_bus;
        end else begin
            bidir_oe_r[KWS_DONE_PAD]   = 1'b1;
            bidir_oe_r[KWS_CLASS0_PAD] = 1'b1;
            bidir_oe_r[KWS_CLASS1_PAD] = 1'b1;
            bidir_oe_r[KWS_CLASS2_PAD] = 1'b1;
            bidir_oe_r[VAD_DROP_PAD]   = 1'b1;

            bidir_out_r[KWS_DONE_PAD]   = kws_done;
            bidir_out_r[KWS_CLASS0_PAD] = kws_class_out[0];
            bidir_out_r[KWS_CLASS1_PAD] = kws_class_out[1];
            bidir_out_r[KWS_CLASS2_PAD] = kws_class_out[2];
            bidir_out_r[VAD_DROP_PAD]   = pipeline_vad_frame_drop;
        end
    end

    // Tie off unused inputs so lint stays clean
    logic _unused;
    // assign _unused = &{bidir_in, input_in[NUM_INPUT_PADS-1:3], 1'b0};
    assign _unused = &{
        bidir_in[NUM_BIDIR_PADS-1:9],
        input_in[NUM_INPUT_PADS-1:9],
        analog,
        1'b0
    };

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

    wire [7:0] tx_byte;
    wire       tx_valid;
    wire       tx_ready;
    wire       tx_busy;

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
        .tx_byte_o       (tx_byte),
        .tx_valid_o      (tx_valid),
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

    // TEST SIGNALS PIPELINE
    logic [CIC_REG_W-1:0]     cic_audio;
    logic signed [FIR_OW-1:0]    FIR_audio_in;
    logic signed [FIR_OW-1:0]    FIR_audio_out;
    logic [ADDR_W-1:0]      sp_a_waddr_test;
    logic signed [7:0]      sp_a_wdata_test;
    logic [ADDR_W-1:0]      sp_b_waddr_test;
    logic signed [7:0]      sp_b_wdata_test;
    logic                   spect_write_sel_test;
    logic [7:0]          test_coeff_addr_i;
    logic [7:0]          test_index_addr_i;
    logic [LUT_FRAC-1:0] test_lut_addr_i;
    logic [1:0] frame_control_state;
    logic [$clog2(N_MELS)-1:0] mel_idx_test;
    logic [POWER_W:0] power_test;

    // TEST SIGNALS KWS
    logic [2:0]fsm_class_test;
    logic  [ADDR_W-1:0]        fsm_a_waddr_test;
    logic  signed [DATA_W-1:0] fsm_a_wdata_test;
    logic  [ADDR_W-1:0]        fsm_a_raddr_test;   
    logic  signed [DATA_W-1:0] fsm_a_rdata_test;
    logic  [ADDR_W-1:0]        fsm_b_waddr_test;
    logic  signed [DATA_W-1:0] fsm_b_wdata_test;
    logic  [ADDR_W-1:0]        fsm_b_raddr_test;   
    logic  signed [DATA_W-1:0] fsm_b_rdata_test; 
    logic  signed [DATA_W-1:0] mac_ifmap_test;
    logic  signed [DATA_W-1:0] mac_weight_test;
    logic  signed [ACC_W-1:0]  mac_bias_test;
    logic  [31:0]              rq_mult_test;
    logic  [4:0]               rq_shift_test;
    logic [3:0]                state_test;
    logic [3:0]                layer_test;
    logic [12:0]          w_raddr_test;
    logic signed [7:0]    w_rdata_test;
    logic [10:0]          ss_a_raddr_test;
    logic signed [7:0]    ss_a_rdata_test;
    logic [10:0]          ss_b_raddr_test;
    logic signed [7:0]    ss_b_rdata_test;
    logic                 mac_en_test;
    logic                 mac_clear_test;
    logic signed [ACC_W-1:0]   acc_test;
    logic                 rq_relu_en_test;
    logic signed [7:0]    rq_out_test;
    logic signed [31:0]   debug_gap0_test;
    logic signed [31:0]   debug_gap1_test;
    logic signed [31:0]   debug_gap2_test;
    logic signed [31:0]   debug_gap3_test;
    logic signed [31:0]   debug_gap4_test;
    logic signed [31:0]   debug_gap5_test;
    logic signed [31:0]   debug_gap6_test;
    logic [8:0]           bias_addr_test;
    logic signed [31:0]   bias_data_test;

    always_comb begin
        dft_debug_bus = '0;

        if (test_mode_audio) begin
            case (dft_sel)
                6'd0: dft_debug_bus = {{(DFT_DEBUG_W-CIC_REG_W){1'b0}}, cic_audio};
                6'd1: dft_debug_bus = {{(DFT_DEBUG_W-FIR_OW){FIR_audio_in[FIR_OW-1]}}, FIR_audio_in};
                6'd2: dft_debug_bus = {{(DFT_DEBUG_W-FIR_OW){FIR_audio_out[FIR_OW-1]}}, FIR_audio_out};
                6'd3:  dft_debug_bus = {{20{1'b0}}, sp_a_waddr_test};
                6'd4:  dft_debug_bus = {{23{sp_a_wdata_test[7]}}, sp_a_wdata_test};
                6'd5:  dft_debug_bus = {{20{1'b0}}, sp_b_waddr_test};
                6'd6:  dft_debug_bus = {{23{sp_b_wdata_test[7]}}, sp_b_wdata_test};
                6'd7:  dft_debug_bus = {{30{1'b0}}, spect_write_sel_test};
                6'd8:  dft_debug_bus = {{23{1'b0}}, test_coeff_addr_i};
                6'd9:  dft_debug_bus = {{23{1'b0}}, test_index_addr_i};
                6'd10: dft_debug_bus = {{(DFT_DEBUG_W-LUT_FRAC){1'b0}}, test_lut_addr_i};
                6'd11: dft_debug_bus = {{29{1'b0}}, frame_control_state};
                6'd12: dft_debug_bus = {{25{1'b0}}, mel_idx_test};
                6'd13: dft_debug_bus = power_test[DFT_DEBUG_W-1:0];
                default: dft_debug_bus = '0;
            endcase
        end else if (test_mode_ml) begin
            case (dft_sel)
                6'd0:  dft_debug_bus = {{28{1'b0}}, fsm_class_test};
                6'd1:  dft_debug_bus = {{20{1'b0}}, fsm_a_waddr_test};
                6'd2:  dft_debug_bus = {{23{fsm_a_wdata_test[7]}}, fsm_a_wdata_test};
                6'd3:  dft_debug_bus = {{20{1'b0}}, fsm_a_raddr_test};
                6'd4:  dft_debug_bus = {{23{fsm_a_rdata_test[7]}}, fsm_a_rdata_test};
                6'd5:  dft_debug_bus = {{20{1'b0}}, fsm_b_waddr_test};
                6'd6:  dft_debug_bus = {{23{fsm_b_wdata_test[7]}}, fsm_b_wdata_test};
                6'd7:  dft_debug_bus = {{20{1'b0}}, fsm_b_raddr_test};
                6'd8:  dft_debug_bus = {{23{fsm_b_rdata_test[7]}}, fsm_b_rdata_test};
                6'd9:  dft_debug_bus = {{23{mac_ifmap_test[7]}}, mac_ifmap_test};
                6'd10: dft_debug_bus = {{23{mac_weight_test[7]}}, mac_weight_test};
                6'd11: dft_debug_bus = mac_bias_test[DFT_DEBUG_W-1:0];
                6'd12: dft_debug_bus = rq_mult_test[DFT_DEBUG_W-1:0];
                6'd13: dft_debug_bus = {{26{1'b0}}, rq_shift_test};
                6'd14: dft_debug_bus = {{27{1'b0}}, state_test};
                6'd15: dft_debug_bus = {{27{1'b0}}, layer_test};
                6'd16: dft_debug_bus = {{18{1'b0}}, w_raddr_test};
                6'd17: dft_debug_bus = {{23{w_rdata_test[7]}}, w_rdata_test};
                6'd18: dft_debug_bus = {{20{1'b0}}, ss_a_raddr_test};
                6'd19: dft_debug_bus = {{23{ss_a_rdata_test[7]}}, ss_a_rdata_test};
                6'd20: dft_debug_bus = {{20{1'b0}}, ss_b_raddr_test};
                6'd21: dft_debug_bus = {{23{ss_b_rdata_test[7]}}, ss_b_rdata_test};
                6'd22: dft_debug_bus = {{30{1'b0}}, mac_en_test};
                6'd23: dft_debug_bus = {{30{1'b0}}, mac_clear_test};
                6'd24: dft_debug_bus = acc_test[DFT_DEBUG_W-1:0];
                6'd25: dft_debug_bus = {{30{1'b0}}, rq_relu_en_test};
                6'd26: dft_debug_bus = {{23{rq_out_test[7]}}, rq_out_test};
                6'd27: dft_debug_bus = debug_gap0_test[DFT_DEBUG_W-1:0];
                6'd28: dft_debug_bus = debug_gap1_test[DFT_DEBUG_W-1:0];
                6'd29: dft_debug_bus = debug_gap2_test[DFT_DEBUG_W-1:0];
                6'd30: dft_debug_bus = debug_gap3_test[DFT_DEBUG_W-1:0];
                6'd31: dft_debug_bus = debug_gap4_test[DFT_DEBUG_W-1:0];
                6'd32: dft_debug_bus = debug_gap5_test[DFT_DEBUG_W-1:0];
                6'd33: dft_debug_bus = debug_gap6_test[DFT_DEBUG_W-1:0];
                6'd34: dft_debug_bus = {{22{1'b0}}, bias_addr_test};
                6'd35: dft_debug_bus = bias_data_test[DFT_DEBUG_W-1:0];
                default: dft_debug_bus = '0;
            endcase
        end
    end

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
        // test signals 
        .test_mode_audio(test_mode_audio),
        .cic_audio(cic_audio),
        .FIR_audio_in(FIR_audio_in),
        .FIR_audio_out(FIR_audio_out),
        .frame_control_state(frame_control_state), 
        .mel_idx_test(mel_idx_test),
        .power_test(power_test),
        .test_coeff_addr_i(test_coeff_addr_i),
        .test_index_addr_i(test_index_addr_i),
        .test_lut_addr_i(test_lut_addr_i),
        .sp_a_waddr_test(sp_a_waddr_test),
        .sp_a_wdata_test(sp_a_wdata_test),
        .sp_b_waddr_test(sp_b_waddr_test),
        .sp_b_wdata_test(sp_b_wdata_test),
        .spect_write_sel_test(spect_write_sel_test)
    );

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

        //test signals 
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
        .layer_test(layer_test),
        .acc_test(acc_test),

        .w_raddr_test(w_raddr_test),
        .w_rdata_test(w_rdata_test),

        .ss_a_raddr_test(ss_a_raddr_test),
        .ss_a_rdata_test(ss_a_rdata_test),
        .ss_b_raddr_test(ss_b_raddr_test),
        .ss_b_rdata_test(ss_b_rdata_test),

        .mac_en_test(mac_en_test),
        .mac_clear_test(mac_clear_test),

        .rq_relu_en_test(rq_relu_en_test),
        .rq_out_test(rq_out_test),

        .debug_gap0_test(debug_gap0_test),
        .debug_gap1_test(debug_gap1_test),
        .debug_gap2_test(debug_gap2_test),
        .debug_gap3_test(debug_gap3_test),
        .debug_gap4_test(debug_gap4_test),
        .debug_gap5_test(debug_gap5_test),
        .debug_gap6_test(debug_gap6_test),
        .bias_addr_test(bias_addr_test),
        .bias_data_test(bias_data_test)
    );

endmodule
`default_nettype wire
