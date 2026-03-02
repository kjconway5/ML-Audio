// Author: Kye Conway

module hpf 
    #(parameter WIDTH = 16) (
      input logic clk,
      input logic reset,
      input logic valid,
      input logic signed [WIDTH-1:0] line_in,
      output logic signed [WIDTH-1:0] line_out
);

    // High-pass filter used to remove low frequencies from the input signal
    // used to remove low rumbles and DC offsets from our input
    // takes in the input audio from an adc and outputs the same signal with low frequencies removed

    // Implement a simple first-order high-pass filter using the difference equation: 
    // out = a * [prev_out + (in - prev_in)]
    // a is approx 1 
    // sum - (sum >>> 6) is approx sum * (63/64)
    // a can be changed to make the filter more or less aggressive

    // MIGHT NEED TO CHANGE BIT WIDTHS TO AVOID OVERFLOW

    logic signed [WIDTH-1:0] prev_line_out;
    logic signed [WIDTH-1:0] prev_line_in;
    logic signed [WIDTH-1:0] diff;
    logic signed [WIDTH-1:0] sum;
    logic signed [WIDTH-1:0] temp_out;

    always_ff @(posedge clk) begin  
        if (reset) begin 
            line_out <= '0;
            prev_line_out <= '0;
            prev_line_in <= '0;
        end else if (valid) begin
            line_out <= temp_out;
            prev_line_out <= temp_out;
            prev_line_in <= line_in;
        end
    end

    always_comb begin
        diff = line_in - prev_line_in;
        sum = prev_line_out + diff;
        temp_out = sum - (sum >>> 6);
    end


endmodule
