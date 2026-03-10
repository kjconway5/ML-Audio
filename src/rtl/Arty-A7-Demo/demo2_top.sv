
module demo2_top (
    input  logic clk_100m,      // 100 MHz oscillator
    input  logic uart_rxd,      // USB-UART RX pin
    output logic uart_txd,      // USB-UART TX pin
    input  logic btn_reset      // Reset button
);

    // prescale = clk_freq / (baud_rate * 8)
    // 100_000_000 / (460_800 * 8) = 27.13
    localparam [15:0] UART_PRESCALE = 16'd27;

   
    logic rst_r, rst_sync;

    always_ff @(posedge clk_100m) begin
        rst_r    <= btn_reset;
        rst_sync <= rst_r;
    end

    // RX side: m_axis = received bytes from host
    logic [7:0] rx_tdata;
    logic       rx_tvalid;
    logic       rx_tready;

    // TX side: s_axis = bytes to send to host
    logic [7:0] tx_tdata;
    logic       tx_tvalid;
    logic       tx_tready;

    uart u_uart (
        .clk             (clk_100m),
        .rst             (rst_sync),
        // TX
        .s_axis_tdata    (tx_tdata),
        .s_axis_tvalid   (tx_tvalid),
        .s_axis_tready   (tx_tready),
        // RX
        .m_axis_tdata    (rx_tdata),
        .m_axis_tvalid   (rx_tvalid),
        .m_axis_tready   (rx_tready),
        // Pins
        .rxd             (uart_rxd),
        .txd             (uart_txd),
        // Status
        .tx_busy         (),
        .rx_busy         (),
        .rx_overrun_error(),
        .rx_frame_error  (),
        // Config
        .prescale        (UART_PRESCALE)
    );

    //Sample unpacker
    logic [13:0] audio_data;
    logic        audio_valid;

    sample_unpacker u_unpack (
        .clk_i         (clk_100m),
        .reset_i       (rst_sync),
        .byte_i        (rx_tdata),
        .byte_valid_i  (rx_tvalid),
        .byte_ready_o  (rx_tready),
        .sample_o      (audio_data),
        .sample_valid_o(audio_valid)
    );

    //Feature extraction pipeline
    logic [15:0] cnn_data;
    logic        cnn_valid;
    logic        cnn_ready;

    pipeline_top u_pipeline (
        .clk_i       (clk_100m),
        .reset_i     (rst_sync),
        .data_i      (audio_data),
        .valid_i     (audio_valid),
        .cnn_data_ol (cnn_data),
        .cnn_valid_ol(cnn_valid),
        .cnn_ready_il(cnn_ready)
    );

    //Packetizer
    packetizer u_pkt (
        .clk_i        (clk_100m),
        .reset_i      (rst_sync),
        .data_i       (cnn_data),
        .valid_i      (cnn_valid),
        .ready_o      (cnn_ready),
        .tx_tdata_o   (tx_tdata),
        .tx_tvalid_o  (tx_tvalid),
        .tx_tready_i  (tx_tready)
    );

endmodule
