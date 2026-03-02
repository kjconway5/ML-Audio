//COUNTER

module example
  #(parameter width_p = 4,

    parameter [width_p-1:0] reset_val_p = '0)
   (input [0:0] clk_i
   ,input [0:0] reset_i
   ,input [0:0] en_i
   ,output logic [width_p-1:0] count_o);

   // Your code here:
always_ff @(posedge clk_i) begin
        if (reset_i) begin
            count_o <= '0;
        end else if (en_i) begin
            count_o <= count_o + 1'b1;
        end 
        
    end
endmodule