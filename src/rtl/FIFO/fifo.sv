module fifo_1r1w
  #(parameter [31:0] width_p = 8,
    // Note: Not depth_p! depth_p should be 1<<depth_log2_p
    /* verilator lint_off WIDTHTRUNC */
    parameter [31:0] depth_log2_p = 8) (
    /* verilator lint_on WIDTHTRUNC */
    input [0:0] clk_i,
    input [0:0] reset_i,
    input [width_p - 1:0] data_i,
    input [0:0] valid_i,
    input [0:0] ready_i,
    output [0:0] ready_o,
    output [0:0] valid_o,
    output [width_p - 1:0] data_o
);

  localparam int depth_p = 1 << depth_log2_p;
  logic [depth_log2_p:0] wr_ptr, rd_ptr;
  logic full, empty;

  ram_1r1w_async #(
    .width_p(width_p),
    .depth_p(depth_p),
    .init_p(1'b0),
    .filename_p("")
  ) ram (
    .clk_i(clk_i),
    .reset_i(reset_i),
    .wr_valid_i(valid_i && !full),
    .wr_data_i(data_i),
    .wr_addr_i(wr_ptr[depth_log2_p-1:0]),
    .rd_addr_i(rd_ptr[depth_log2_p-1:0]),
    .rd_data_o(data_o)
  );

  assign empty = (wr_ptr == rd_ptr); // if the cur addr is =, must be empty
  // if the next pointer is at read, must be full, if depth_log2 is 1, uses an extra bit otherwise the FIFO is always true
  // because it can only be 1 or 0
  assign full = ((wr_ptr[depth_log2_p-1:0] == rd_ptr[depth_log2_p-1:0]) && (wr_ptr[depth_log2_p] != rd_ptr[depth_log2_p])); 
  assign ready_o = !full; // if FIFO not full, stay in IDLE to write
  assign valid_o = !empty; // if FIFO is full, move to IHVD to read

  always_ff @(posedge clk_i) begin
    if (reset_i) begin // start pointers at address 0
      wr_ptr <= '0;
      rd_ptr <= '0;
    end else begin
      // read and write in seperate ifs to allow read and write in same clock cycle 
      if (valid_i && ready_o) begin // IDLE: move write pointer by one
        wr_ptr <= (wr_ptr + 1'b1);
      end 
      if (valid_o && ready_i) begin // IHVD: move read ptr by one
        rd_ptr <= (rd_ptr + 1'b1);
      end
    end
  end

endmodule
