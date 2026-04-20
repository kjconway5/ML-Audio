
// One 8-bit-wide x 12-deep SRAM. While `flash_en_i` is held high,
// every received UART byte is written into the next address. When the
// counter reaches DEPTH-1 it stops; releasing `flash_en_i` rewinds it
// back to zero.
module simple_flash #(
    parameter DEPTH       = 12,
    parameter CLK_FREQ_HZ = 12_000_000,    // iCE40 clock
    parameter BAUD        = 115_200,
    parameter PRESCALE    = (CLK_FREQ_HZ / BAUD) / 8
)(
    input  logic                       clk_i,
    input  logic                       reset_i,
    // UART pins
    input  logic                       uart_rxd,
    output logic                       uart_txd,
    // Mode pin -- strap high while host is sending bytes
    input  logic                       flash_en_i,
    // Read port
    input  logic [$clog2(DEPTH)-1:0]   rd_addr_i,
    output logic [7:0]                 rd_data_o
);


    // UART receiver
    logic [7:0] rx_data;
    logic       rx_valid;
    logic       rx_ready;

    uart_rx #(
        .DATA_WIDTH (8)
    ) u_uart_rx (
        .clk           (clk_i),
        .rst           (reset_i),
        .m_axis_tdata  (rx_data),
        .m_axis_tvalid (rx_valid),
        .m_axis_tready (rx_ready),
        .rxd           (uart_rxd),
        .busy          (),
        .overrun_error (),
        .frame_error   (),
        .prescale      (PRESCALE[15:0])
    );

    // No TX 
    assign uart_txd = 1'b1;

    //Flash Controller
    logic [$clog2(DEPTH)-1:0] addr_q;
    logic                     full_q;     
    logic                     wr_valid;

    // Accept a byte only while the button is held AND we still have room.
    assign rx_ready = flash_en_i && !full_q;
    assign wr_valid = rx_ready && rx_valid;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            addr_q <= '0;
            full_q <= 1'b0;
        end else if (!flash_en_i) begin
            // Button released
            addr_q <= '0;
            full_q <= 1'b0;
        end else if (wr_valid) begin
            if (addr_q == DEPTH[$clog2(DEPTH)-1:0] - 1'b1) begin
                full_q <= 1'b1;
            end else begin
                addr_q <= addr_q + 1'b1;
            end
        end
    end

    //"SRAM-MACRO"
    ram_1r1w_sync #(
        .width_p (8),
        .depth_p (DEPTH)
    ) u_ram (
        .clk_i      (clk_i),
        .reset_i    (reset_i),

        .wr_valid_i (wr_valid),
        .wr_data_i  (rx_data),
        .wr_addr_i  (addr_q),

        .rd_valid_i (1'b1),
        .rd_addr_i  (rd_addr_i),
        .rd_data_o  (rd_data_o)
    );

endmodule
