module counter
  #(parameter [31:0] max_val_p = 15
   ,parameter width_p = $clog2(max_val_p)  
    /* verilator lint_off WIDTHTRUNC */
   ,parameter [width_p-1:0] reset_val_p = '0
    )
    /* verilator lint_on WIDTHTRUNC */
   (input [0:0] clk_i
   ,input [0:0] reset_i
   ,input [0:0] up_i
   ,input [0:0] down_i
   ,output [width_p-1:0] count_o);

  localparam [width_p-1:0] max_val_lp = max_val_p[width_p-1:0];

  logic [width_p-1:0] count_d, count_q;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      count_q <= reset_val_p;
    end else begin
      count_q <= count_d;
    end
  end

  always_comb begin
    count_d = count_q;
    if (up_i & !down_i) begin
      if (count_q == max_val_lp ) begin
        count_d = '0;
      end else begin
        count_d= count_q + 1;
      end
    end else if (down_i & !up_i) begin
      if (count_q == 0) begin
        count_d = max_val_lp;
      end else begin
        count_d= count_q - 1;
      end

    end else begin
      count_d = count_q;
    end
  end

  assign count_o = count_d;
  
endmodule