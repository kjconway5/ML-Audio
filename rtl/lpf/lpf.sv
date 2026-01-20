module lpf 
    #(parameter WIDTH = 16) (
      input logic clk,
      input logic reset,
      input logic valid,
      input logic signed [WIDTH-1:0] line_in,
      output logic signed [WIDTH-1:0] line_out
);

    // Low-pass filter used to remove high frequencies from the input signal
    // used to remove hissy sounds and high frequency noise from our input
    // takes in the input audio from an adc and outputs the same signal with high frequencies removed

    // Implement a simple first-order low-pass filter using the difference equation: 
    // diff = line_in - prev_line_out
	// scaled = a * diff
	// y = prev_line_out + scaled
	// store prev_line_out <= line_out
    // a = 1/8

    // line_out = prev_line_out + a * (line_in - prev_line_out)

    // MIGHT NEED TO CHANGE BIT WIDTHS TO AVOID OVERFLOW

    logic signed [WIDTH-1:0] prev_line_out;
    logic signed [WIDTH-1:0] diff;
    logic signed [WIDTH-1:0] scaled;
    logic signed [WIDTH-1:0] temp_out;

    always_ff @(posedge clk) begin  
        if (reset) begin 
            line_out <= '0;
            prev_line_out <= '0;
        end else if (valid) begin
            line_out <= temp_out;
            prev_line_out <= temp_out;
        end
    end

    always_comb begin
        diff = line_in - prev_line_out;
        scaled = diff >>> 3;
        temp_out = prev_line_out + scaled;
    end

endmodule
