module bias_DFFs #(
    parameter DEPTH  = 295,   // 9x32 channels + 7 classifier = 295 (32-filter model)
    parameter DATA_W = 32,
    parameter ADDR_W = 9      // 9-bit: supports bias offsets up to 288
)(
    input  wire [ADDR_W-1:0]        addr,
    output reg  signed [DATA_W-1:0] data
);
    always @(*) begin
        case (addr)
            // first_conv (bias_off=0, 16 channels)
            9'd0: data = 32'sh00003747;
            9'd1: data = 32'shFFFFCBA8;
            9'd2: data = 32'sh00002797;
            9'd3: data = 32'sh000020C9;
            9'd4: data = 32'shFFFFD247;
            9'd5: data = 32'shFFFFCAF7;
            9'd6: data = 32'sh00002CC7;
            9'd7: data = 32'shFFFFBB18;
            9'd8: data = 32'shFFFFE85A;
            9'd9: data = 32'sh00002061;
            9'd10: data = 32'shFFFFD1BA;
            9'd11: data = 32'shFFFFB9FE;
            9'd12: data = 32'sh000025A3;
            9'd13: data = 32'sh00001D8A;
            9'd14: data = 32'shFFFFD665;
            9'd15: data = 32'shFFFFCE0A;

            // ds_blocks.0.depthwise (bias_off=16, 16 channels)
            9'd16: data = 32'sh000003AB;
            9'd17: data = 32'sh0000025C;
            9'd18: data = 32'sh00000D85;
            9'd19: data = 32'shFFFFFC5E;
            9'd20: data = 32'sh00000084;
            9'd21: data = 32'sh00000264;
            9'd22: data = 32'sh00000595;
            9'd23: data = 32'sh000002C7;
            9'd24: data = 32'shFFFFF9ED;
            9'd25: data = 32'sh00000348;
            9'd26: data = 32'shFFFFFE2B;
            9'd27: data = 32'sh000001A8;
            9'd28: data = 32'shFFFFFE1C;
            9'd29: data = 32'sh00000430;
            9'd30: data = 32'shFFFFFDD1;
            9'd31: data = 32'sh0000013B;

            // ds_blocks.0.pointwise (bias_off=32, 16 channels)
            9'd32: data = 32'shFFFFFC27;
            9'd33: data = 32'shFFFFFFC4;
            9'd34: data = 32'shFFFFFEFB;
            9'd35: data = 32'sh0000003C;
            9'd36: data = 32'shFFFFFB0F;
            9'd37: data = 32'sh000002B6;
            9'd38: data = 32'shFFFFFEAA;
            9'd39: data = 32'sh00000446;
            9'd40: data = 32'sh0000057F;
            9'd41: data = 32'shFFFFFF76;
            9'd42: data = 32'shFFFFFE53;
            9'd43: data = 32'shFFFFFD9D;
            9'd44: data = 32'sh00000689;
            9'd45: data = 32'sh000002AB;
            9'd46: data = 32'shFFFFFDB9;
            9'd47: data = 32'sh000000C9;

            // ds_blocks.1.depthwise (bias_off=48, 16 channels)
            9'd48: data = 32'shFFFFFFC7;
            9'd49: data = 32'shFFFFFF70;
            9'd50: data = 32'shFFFFFEFF;
            9'd51: data = 32'shFFFFFFE8;
            9'd52: data = 32'sh00000416;
            9'd53: data = 32'shFFFFFAEC;
            9'd54: data = 32'sh0000033D;
            9'd55: data = 32'sh000003BF;
            9'd56: data = 32'shFFFFF87C;
            9'd57: data = 32'shFFFFFEEC;
            9'd58: data = 32'sh0000007E;
            9'd59: data = 32'sh00000262;
            9'd60: data = 32'shFFFFFD4E;
            9'd61: data = 32'shFFFFFECB;
            9'd62: data = 32'sh00000256;
            9'd63: data = 32'shFFFFFFD6;

            // ds_blocks.1.pointwise (bias_off=64, 16 channels)
            9'd64: data = 32'sh0000081D;
            9'd65: data = 32'sh00000158;
            9'd66: data = 32'shFFFFFC24;
            9'd67: data = 32'sh0000106F;
            9'd68: data = 32'sh00000166;
            9'd69: data = 32'sh000003CE;
            9'd70: data = 32'shFFFFFA1C;
            9'd71: data = 32'shFFFFFFA0;
            9'd72: data = 32'shFFFFFD1B;
            9'd73: data = 32'sh00000037;
            9'd74: data = 32'sh000007CB;
            9'd75: data = 32'sh000004E6;
            9'd76: data = 32'sh00000BDD;
            9'd77: data = 32'sh000001CC;
            9'd78: data = 32'sh000008DB;
            9'd79: data = 32'sh000004EA;

            // ds_blocks.2.depthwise (bias_off=80, 16 channels)
            9'd80: data = 32'shFFFFF642;
            9'd81: data = 32'shFFFFF964;
            9'd82: data = 32'sh00000E3A;
            9'd83: data = 32'shFFFFF8C9;
            9'd84: data = 32'shFFFFF614;
            9'd85: data = 32'sh00000A86;
            9'd86: data = 32'sh00000778;
            9'd87: data = 32'shFFFFFE54;
            9'd88: data = 32'sh000002AC;
            9'd89: data = 32'shFFFFF779;
            9'd90: data = 32'shFFFFFD77;
            9'd91: data = 32'shFFFFF0E6;
            9'd92: data = 32'shFFFFEF9B;
            9'd93: data = 32'shFFFFFB32;
            9'd94: data = 32'sh00000D8C;
            9'd95: data = 32'sh00000825;

            // ds_blocks.2.pointwise (bias_off=96, 16 channels)
            9'd96: data = 32'shFFFFFB86;
            9'd97: data = 32'shFFFFFEAF;
            9'd98: data = 32'shFFFFFB25;
            9'd99: data = 32'shFFFFFDC2;
            9'd100: data = 32'shFFFFFE73;
            9'd101: data = 32'shFFFFF659;
            9'd102: data = 32'shFFFFF867;
            9'd103: data = 32'shFFFFF84A;
            9'd104: data = 32'shFFFFF7C8;
            9'd105: data = 32'shFFFFFFAD;
            9'd106: data = 32'shFFFFFDF2;
            9'd107: data = 32'sh000000BF;
            9'd108: data = 32'sh000000C4;
            9'd109: data = 32'sh0000026E;
            9'd110: data = 32'shFFFFFA3B;
            9'd111: data = 32'sh0000044F;

            // ds_blocks.3.depthwise (bias_off=112, 16 channels)
            9'd112: data = 32'shFFFFF6C7;
            9'd113: data = 32'sh000003AC;
            9'd114: data = 32'sh0000043D;
            9'd115: data = 32'sh00000388;
            9'd116: data = 32'sh0000043A;
            9'd117: data = 32'sh000005CF;
            9'd118: data = 32'sh00000678;
            9'd119: data = 32'sh00000468;
            9'd120: data = 32'shFFFFF743;
            9'd121: data = 32'shFFFFF947;
            9'd122: data = 32'shFFFFFA53;
            9'd123: data = 32'shFFFFF4D3;
            9'd124: data = 32'sh000004BD;
            9'd125: data = 32'shFFFFF961;
            9'd126: data = 32'shFFFFFD9A;
            9'd127: data = 32'shFFFFF873;

            // ds_blocks.3.pointwise (bias_off=128, 16 channels)
            9'd128: data = 32'sh00000225;
            9'd129: data = 32'shFFFFFEB0;
            9'd130: data = 32'sh000000BC;
            9'd131: data = 32'sh00000091;
            9'd132: data = 32'shFFFFFFE6;
            9'd133: data = 32'sh0000028B;
            9'd134: data = 32'sh000001E1;
            9'd135: data = 32'sh000001A1;
            9'd136: data = 32'shFFFFFECA;
            9'd137: data = 32'sh00000139;
            9'd138: data = 32'sh00000127;
            9'd139: data = 32'sh000001EC;
            9'd140: data = 32'shFFFFFECF;
            9'd141: data = 32'shFFFFFFE8;
            9'd142: data = 32'sh00000082;
            9'd143: data = 32'sh00000069;

            // classifier (bias_off=144, 7 channels)
            9'd144: data = 32'sh00000018;
            9'd145: data = 32'sh00000002;
            9'd146: data = 32'sh00000026;
            9'd147: data = 32'shFFFFFFD5;
            9'd148: data = 32'sh00000014;
            9'd149: data = 32'shFFFFFFED;
            9'd150: data = 32'shFFFFFFFC;

            default: data = 32'sh00000000;
        endcase
    end

endmodule
