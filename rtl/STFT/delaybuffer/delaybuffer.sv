module delaybuffer
  #(parameter [31:0] width_p = 8
   ,parameter [31:0] delay_p = 8
   )
  (input [0:0] clk_i
  ,input [0:0] reset_i

  ,input [width_p - 1:0] data_i
  ,input [0:0] valid_i
  ,output [0:0] ready_o 

  ,output logic [0:0] valid_o 
  ,output [width_p - 1:0] data_o 
  ,input [0:0] ready_i
  );

  localparam width_lp = (delay_p > 1) ? $clog2(delay_p) : 1;

  logic [$clog2(delay_p)-1:0] wr_rd_ptr;
  logic valid_r;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      valid_r <= '0;
    end else if (ready_o) begin
      valid_r <= valid_i;
    end
  end

  assign ready_o = !valid_o | ready_i;
  assign valid_o = valid_r;



  logic [$clog2(delay_p)-1:0] count_d, count_q;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      count_q <= '0;
    end else begin
      count_q <= count_d;
    end
  end


  always_comb begin
    count_d = count_q;
    if (ready_i & valid_o) begin
      if (count_q == delay_p[width_lp-1:0] - 1) begin
        count_d = '0;
      end else begin
        count_d = count_q + 1;
      end
    end else begin
      count_d = count_q;
    end
  end

  assign wr_rd_ptr = count_d;


  ram_1r1w_sync #(
    .width_p(width_p),
    .depth_p(delay_p),
    .filename_p()
  )
  ram_1r1w_sync_inst(
    .clk_i(clk_i),
    .reset_i(reset_i),
    .wr_valid_i(ready_o & valid_i), // valid_i ready_o
    .wr_addr_i(wr_rd_ptr),
    .wr_data_i(data_i),
    .rd_valid_i(ready_i), // ready_i valid_o
    .rd_addr_i(wr_rd_ptr),
    .rd_data_o(data_o)
  );



endmodule