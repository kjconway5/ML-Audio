// mac_array.sv
// Scalar MAC accumulator: one INT8 × INT8 multiply-accumulate per clock cycle.

module mac_array #(
    parameter DATA_W = 8,
    parameter ACC_W  = 32
)(
    input  wire                      clk,
    input  wire                      reset,
    input  wire                      en,
    input  wire                      clear,
    input  wire signed [DATA_W-1:0]  ifmap,
    input  wire signed [DATA_W-1:0]  weight,
    input  wire signed [ACC_W-1:0]   bias,
    output reg  signed [ACC_W-1:0]   acc
);

    always @(posedge clk) begin
        if (reset) begin
            acc <= 0;
        end else if (clear) begin
            acc <= bias;
        end else if (en) begin
            acc <= acc + $signed({{(ACC_W-DATA_W){ifmap[DATA_W-1]}},  ifmap})
                       * $signed({{(ACC_W-DATA_W){weight[DATA_W-1]}}, weight});
        end
    end

endmodule
