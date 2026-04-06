module mac_array #(
    parameter N_MACS = 16,
    parameter DATA_W = 8,
    parameter ACC_W  = 32
)(
    input  wire                      clk,
    input  wire                      reset,
    input  wire                      en,        
    input  wire                      clear,    
    input  wire signed [DATA_W-1:0]  ifmap  [0:N_MACS-1],
    input  wire signed [DATA_W-1:0]  weight [0:N_MACS-1],
    input  wire signed [ACC_W-1:0]   bias,
    output reg  signed [ACC_W-1:0]   acc,
    output reg                       valid
);
    integer i;
    reg signed [ACC_W-1:0] sum;
    always @(posedge clk) begin
        if (reset) begin
            acc   <= 0;
        end else if (clear) begin
            acc   <= bias;
        end else if (en) begin
            sum = 0;
            for (i = 0; i < N_MACS; i = i + 1)
                sum = sum + (ifmap[i] * weight[i]);
            acc <= sum;
        end 
        // else begin
        //     acc <= acc; // hold value
        // end
    end

    assign valid = en ? 1 : 0;

endmodule