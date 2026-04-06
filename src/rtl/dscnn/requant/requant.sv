module requant #(
    parameter ACC_W  = 32,
    parameter DATA_W = 8
)(
    input  wire signed [ACC_W-1:0]   acc,
    input  wire        [4:0]         shift,    // shift value based on size after MAC (more computations -> more shift) 
    input  wire                      relu_en,  // 1 for all layers except classifier
    output wire signed [DATA_W-1:0]  out
);
    wire signed [ACC_W-1:0] shifted = acc >>> shift;  // shift magnitude so value fits in 8-bits 

    
    wire signed [DATA_W-1:0] saturated =            // Truncate 32 -> 8 bits 
        (shifted > 32'sh0000007F) ?  8'sh7F :       
        (shifted < -32'sh00000080) ? -8'sh80 :
        shifted[DATA_W-1:0];

    // If saturated (is negative), perform Relu which sets negative numbers to 0 
    assign out = (relu_en && saturated[DATA_W-1]) ? 8'sh00 : saturated;

endmodule
