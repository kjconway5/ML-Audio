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
    always @(posedge clk) begin
        if (reset) begin
            acc   <= 0;
            valid <= 0;
        end else if (clear) begin
            acc   <= bias;
            valid <= 0;
        end else if (en) begin
            for (i = 0; i < N_MACS; i = i + 1)
                acc <= acc + (ifmap[i] * weight[i]);
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end
endmodule