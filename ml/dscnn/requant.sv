module requant #(
    parameter ACC_W  = 32,
    parameter DATA_W = 8
)(
    input  wire signed [ACC_W-1:0]   acc,
    input  wire        [4:0]         shift,    // per-layer from controller ROM
    input  wire                      relu_en,  // 1 for all layers except classifier
    output wire signed [DATA_W-1:0]  out
);
    wire signed [ACC_W-1:0] shifted = acc >>> shift;

    
    wire signed [DATA_W-1:0] saturated =            //If still out of range after shift, perform saturation 
        (shifted > 32'sh0000007F) ?  8'sh7F :       //Set values to 127 or -128 
        (shifted < -32'sh00000080) ? -8'sh80 :
        shifted[DATA_W-1:0];

    //If saturated is negative (top bit 1) perform Relu which sets to negative numbers to 0 
    assign out = (relu_en && saturated[DATA_W-1]) ? 8'sh00 : saturated;

endmodule
