module bias_DFFs #(
    parameter DEPTH  = 295,   // 9×32 channels + 7 classifier = 295 (32-filter model)
    parameter DATA_W = 32,
    parameter ADDR_W = 9      // 9-bit: supports bias offsets 256 and 288 in 32-filter model
)(
    input  wire [ADDR_W-1:0]        addr,
    output reg  signed [DATA_W-1:0] data
);
    always @(*) begin
        case (addr)
            // first_conv (bias_off=0, 16 channels)
            9'd0: data = 32'shFFFFEA0E;
            9'd1: data = 32'shFFFFFF63;
            9'd2: data = 32'shFFFFE164;
            9'd3: data = 32'sh00004219;
            9'd4: data = 32'shFFFFEAE3;
            9'd5: data = 32'shFFFFBD1B;
            9'd6: data = 32'sh000009F8;
            9'd7: data = 32'sh00003D19;
            9'd8: data = 32'sh000002E1;
            9'd9: data = 32'sh00003139;
            9'd10: data = 32'shFFFFFED9;
            9'd11: data = 32'sh00003074;
            9'd12: data = 32'shFFFFD1BD;
            9'd13: data = 32'sh000031F2;
            9'd14: data = 32'sh00001686;
            9'd15: data = 32'sh00003FBF;

            // ds_blocks.0.depthwise (bias_off=16, 16 channels)
            9'd16: data = 32'shFFFFFE15;
            9'd17: data = 32'sh000000EC;
            9'd18: data = 32'sh000000F7;
            9'd19: data = 32'shFFFFFD30;
            9'd20: data = 32'sh00000758;
            9'd21: data = 32'shFFFFFFFD;
            9'd22: data = 32'sh00000698;
            9'd23: data = 32'shFFFFFAB6;
            9'd24: data = 32'shFFFFFFDA;
            9'd25: data = 32'shFFFFFC63;
            9'd26: data = 32'shFFFFFD7F;
            9'd27: data = 32'sh00000456;
            9'd28: data = 32'shFFFFFFA5;
            9'd29: data = 32'sh0000056B;
            9'd30: data = 32'shFFFFFF30;
            9'd31: data = 32'shFFFFFC86;

            // ds_blocks.0.pointwise (bias_off=32, 16 channels)
            9'd32: data = 32'sh00000213;
            9'd33: data = 32'sh00000275;
            9'd34: data = 32'shFFFFF9BE;
            9'd35: data = 32'sh000004E0;
            9'd36: data = 32'shFFFFFDFA;
            9'd37: data = 32'sh00000603;
            9'd38: data = 32'sh000003DA;
            9'd39: data = 32'shFFFFFDC6;
            9'd40: data = 32'sh000000B1;
            9'd41: data = 32'sh0000016D;
            9'd42: data = 32'sh0000048F;
            9'd43: data = 32'sh00000016;
            9'd44: data = 32'shFFFFFF7B;
            9'd45: data = 32'sh000007E9;
            9'd46: data = 32'sh00000577;
            9'd47: data = 32'shFFFFF9EB;

            // ds_blocks.1.depthwise (bias_off=48, 16 channels)
            9'd48: data = 32'sh0000038C;
            9'd49: data = 32'sh000003E4;
            9'd50: data = 32'shFFFFFE3F;
            9'd51: data = 32'shFFFFFFF8;
            9'd52: data = 32'shFFFFFDA5;
            9'd53: data = 32'shFFFFFA82;
            9'd54: data = 32'shFFFFFEEC;
            9'd55: data = 32'shFFFFFFFF;
            9'd56: data = 32'sh0000014C;
            9'd57: data = 32'sh000003AD;
            9'd58: data = 32'sh00000597;
            9'd59: data = 32'shFFFFFD1D;
            9'd60: data = 32'sh000002AB;
            9'd61: data = 32'shFFFFFFE7;
            9'd62: data = 32'shFFFFFF05;
            9'd63: data = 32'sh00000136;

            // ds_blocks.1.pointwise (bias_off=64, 16 channels)
            9'd64: data = 32'sh00000171;
            9'd65: data = 32'sh00000B78;
            9'd66: data = 32'shFFFFFADA;
            9'd67: data = 32'shFFFFFC88;
            9'd68: data = 32'shFFFFFB1B;
            9'd69: data = 32'sh00000317;
            9'd70: data = 32'shFFFFFF18;
            9'd71: data = 32'sh00000382;
            9'd72: data = 32'shFFFFFE86;
            9'd73: data = 32'sh000005E6;
            9'd74: data = 32'shFFFFF287;
            9'd75: data = 32'sh000008EC;
            9'd76: data = 32'sh000005D1;
            9'd77: data = 32'sh000005C5;
            9'd78: data = 32'sh00000731;
            9'd79: data = 32'sh00000338;

            // ds_blocks.2.depthwise (bias_off=80, 16 channels)
            9'd80: data = 32'sh000007A2;
            9'd81: data = 32'shFFFFFC17;
            9'd82: data = 32'shFFFFFE03;
            9'd83: data = 32'sh00000526;
            9'd84: data = 32'sh00000936;
            9'd85: data = 32'sh00000289;
            9'd86: data = 32'sh00000400;
            9'd87: data = 32'sh00000784;
            9'd88: data = 32'sh000006F7;
            9'd89: data = 32'sh000006B8;
            9'd90: data = 32'shFFFFFAD0;
            9'd91: data = 32'sh00000197;
            9'd92: data = 32'sh00000BBA;
            9'd93: data = 32'shFFFFF95F;
            9'd94: data = 32'sh00000982;
            9'd95: data = 32'shFFFFF6E4;

            // ds_blocks.2.pointwise (bias_off=96, 16 channels)
            9'd96: data = 32'shFFFFFD19;
            9'd97: data = 32'shFFFFFCDF;
            9'd98: data = 32'shFFFFFF34;
            9'd99: data = 32'shFFFFFF0D;
            9'd100: data = 32'shFFFFF9C3;
            9'd101: data = 32'sh0000005F;
            9'd102: data = 32'shFFFFFFC6;
            9'd103: data = 32'shFFFFFFF8;
            9'd104: data = 32'shFFFFFC46;
            9'd105: data = 32'shFFFFFE53;
            9'd106: data = 32'sh00000457;
            9'd107: data = 32'shFFFFFF2C;
            9'd108: data = 32'sh0000009F;
            9'd109: data = 32'shFFFFFD83;
            9'd110: data = 32'shFFFFFB8E;
            9'd111: data = 32'shFFFFFF46;

            // ds_blocks.3.depthwise (bias_off=112, 16 channels)
            9'd112: data = 32'shFFFFFF91;
            9'd113: data = 32'sh000000E5;
            9'd114: data = 32'sh00000174;
            9'd115: data = 32'sh0000002F;
            9'd116: data = 32'shFFFFFF1D;
            9'd117: data = 32'shFFFFFF4B;
            9'd118: data = 32'sh000000F5;
            9'd119: data = 32'sh0000007D;
            9'd120: data = 32'shFFFFFFA5;
            9'd121: data = 32'sh00000112;
            9'd122: data = 32'sh00000130;
            9'd123: data = 32'shFFFFFF34;
            9'd124: data = 32'shFFFFFFCE;
            9'd125: data = 32'sh000000F0;
            9'd126: data = 32'shFFFFFFA4;
            9'd127: data = 32'shFFFFFF5A;

            // ds_blocks.3.pointwise (bias_off=128, 16 channels)
            9'd128: data = 32'sh00000005;
            9'd129: data = 32'sh00000093;
            9'd130: data = 32'sh00000165;
            9'd131: data = 32'shFFFFFFD6;
            9'd132: data = 32'shFFFFFFFE;
            9'd133: data = 32'sh00000073;
            9'd134: data = 32'sh0000005F;
            9'd135: data = 32'shFFFFFFB0;
            9'd136: data = 32'sh000000D1;
            9'd137: data = 32'sh000000A0;
            9'd138: data = 32'sh00000022;
            9'd139: data = 32'shFFFFFFDC;
            9'd140: data = 32'sh0000005A;
            9'd141: data = 32'sh000000BF;
            9'd142: data = 32'shFFFFFF6F;
            9'd143: data = 32'shFFFFFF69;

            // classifier (bias_off=144, 7 channels)
            9'd144: data = 32'shFFFFFFF6;
            9'd145: data = 32'sh00000028;
            9'd146: data = 32'shFFFFFFE8;
            9'd147: data = 32'sh00000038;
            9'd148: data = 32'shFFFFFFDC;
            9'd149: data = 32'shFFFFFFF2;
            9'd150: data = 32'shFFFFFFED;

            default: data = 32'sh00000000;
        endcase
    end

endmodule
