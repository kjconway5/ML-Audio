module bias_DFFs #(
    parameter DEPTH  = 223,
    parameter DATA_W = 32,
    parameter ADDR_W = 8
)(
    input  wire [ADDR_W-1:0]        addr,   // no clk — purely combinational
    output reg  signed [DATA_W-1:0] data    // reg because driven by always@(*)
);
    always @(*) begin
        case (addr)
            // first_conv (offset 0, 24 values)
            8'd0:  data = 32'sh0005DAB4;
            8'd1:  data = 32'shFFFB6A5B;
            8'd2:  data = 32'sh0003D0CB;
            8'd3:  data = 32'shFFFFF589;
            8'd4:  data = 32'sh0004EEF5;
            8'd5:  data = 32'sh0005A169;
            8'd6:  data = 32'sh000585D1;
            8'd7:  data = 32'shFFFB1AF9;
            8'd8:  data = 32'sh000555B6;
            8'd9:  data = 32'shFFFA892C;
            8'd10: data = 32'sh0002E223;
            8'd11: data = 32'sh000374CA;
            8'd12: data = 32'shFFFBFC59;
            8'd13: data = 32'sh000285CC;
            8'd14: data = 32'shFFFC43A9;
            8'd15: data = 32'shFFFC35E4;
            8'd16: data = 32'sh00036189;
            8'd17: data = 32'sh00039109;
            8'd18: data = 32'sh0003A74F;
            8'd19: data = 32'sh0001F4B3;
            8'd20: data = 32'sh0005E39D;
            8'd21: data = 32'sh0003BFE0;
            8'd22: data = 32'sh00004B93;
            8'd23: data = 32'sh0009A8FD;

            // ds_blocks.0.depthwise (offset 24, 24 values)
            8'd24: data = 32'shFFFFFDD1;
            8'd25: data = 32'shFFFFFD37;
            8'd26: data = 32'sh0000028A;
            8'd27: data = 32'sh0000054F;
            8'd28: data = 32'sh00000405;
            8'd29: data = 32'shFFFFFDB8;
            8'd30: data = 32'sh0000077E;
            8'd31: data = 32'sh0000046B;
            8'd32: data = 32'sh00000775;
            8'd33: data = 32'sh0000030B;
            8'd34: data = 32'shFFFFFE6D;
            8'd35: data = 32'sh0000152A;
            8'd36: data = 32'sh00000047;
            8'd37: data = 32'sh000002CE;
            8'd38: data = 32'shFFFFFD08;
            8'd39: data = 32'shFFFFFE0A;
            8'd40: data = 32'sh0000093B;
            8'd41: data = 32'sh00000357;
            8'd42: data = 32'shFFFFFD81;
            8'd43: data = 32'sh000001E8;
            8'd44: data = 32'sh000003E5;
            8'd45: data = 32'sh00000A51;
            8'd46: data = 32'sh000003E1;
            8'd47: data = 32'sh00000386;

            // ds_blocks.0.pointwise (offset 48, 24 values)
            8'd48: data = 32'shFFFFFD7B;
            8'd49: data = 32'shFFFFDCBB;
            8'd50: data = 32'sh0000005D;
            8'd51: data = 32'sh00000411;
            8'd52: data = 32'sh000010F9;
            8'd53: data = 32'shFFFFE9E5;
            8'd54: data = 32'shFFFFED73;
            8'd55: data = 32'sh00002A76;
            8'd56: data = 32'shFFFFE866;
            8'd57: data = 32'sh00001BF1;
            8'd58: data = 32'sh0000241B;
            8'd59: data = 32'sh00001402;
            8'd60: data = 32'sh0000003C;
            8'd61: data = 32'sh000001BD;
            8'd62: data = 32'sh00000903;
            8'd63: data = 32'sh00002096;
            8'd64: data = 32'sh0000043B;
            8'd65: data = 32'sh00001981;
            8'd66: data = 32'sh00002FE3;
            8'd67: data = 32'sh00002091;
            8'd68: data = 32'sh00003437;
            8'd69: data = 32'sh00001012;
            8'd70: data = 32'sh000011C0;
            8'd71: data = 32'shFFFFF55C;

            // ds_blocks.1.depthwise (offset 72, 24 values)
            8'd72: data = 32'shFFFFFD1A;
            8'd73: data = 32'sh00000572;
            8'd74: data = 32'shFFFFFEDD;
            8'd75: data = 32'sh000004EA;
            8'd76: data = 32'shFFFFFD1F;
            8'd77: data = 32'sh0000064A;
            8'd78: data = 32'shFFFFFB7A;
            8'd79: data = 32'shFFFFFE55;
            8'd80: data = 32'sh00000635;
            8'd81: data = 32'shFFFFF938;
            8'd82: data = 32'sh0000062D;
            8'd83: data = 32'shFFFFFF8C;
            8'd84: data = 32'sh0000055D;
            8'd85: data = 32'shFFFFFF77;
            8'd86: data = 32'sh00000614;
            8'd87: data = 32'shFFFFFEB6;
            8'd88: data = 32'sh00000085;
            8'd89: data = 32'sh00000452;
            8'd90: data = 32'sh00000127;
            8'd91: data = 32'shFFFFFFEE;
            8'd92: data = 32'sh000008CF;
            8'd93: data = 32'sh00000460;
            8'd94: data = 32'sh0000046B;
            8'd95: data = 32'sh00000004;

            // ds_blocks.1.pointwise (offset 96, 24 values)
            8'd96:  data = 32'shFFFFFCCA;
            8'd97:  data = 32'sh00001B43;
            8'd98:  data = 32'sh0000034F;
            8'd99:  data = 32'sh00001192;
            8'd100: data = 32'sh000003B3;
            8'd101: data = 32'shFFFFFFD8;
            8'd102: data = 32'sh00000134;
            8'd103: data = 32'sh00000FF7;
            8'd104: data = 32'shFFFFE8B0;
            8'd105: data = 32'sh000005CE;
            8'd106: data = 32'sh000003FC;
            8'd107: data = 32'sh000009ED;
            8'd108: data = 32'sh00000ED8;
            8'd109: data = 32'sh000008C2;
            8'd110: data = 32'shFFFFEDAB;
            8'd111: data = 32'sh00000387;
            8'd112: data = 32'sh000019A0;
            8'd113: data = 32'shFFFFE785;
            8'd114: data = 32'sh00000588;
            8'd115: data = 32'shFFFFFF57;
            8'd116: data = 32'sh0000029A;
            8'd117: data = 32'sh00000F20;
            8'd118: data = 32'sh000007E7;
            8'd119: data = 32'sh000000F0;

            // ds_blocks.2.depthwise (offset 120, 24 values)
            8'd120: data = 32'shFFFFFDA1;
            8'd121: data = 32'shFFFFFC65;
            8'd122: data = 32'shFFFFFF8E;
            8'd123: data = 32'sh00000299;
            8'd124: data = 32'sh0000023A;
            8'd125: data = 32'sh000003A0;
            8'd126: data = 32'shFFFFFEF4;
            8'd127: data = 32'sh0000064B;
            8'd128: data = 32'shFFFFFDE5;
            8'd129: data = 32'sh000004CA;
            8'd130: data = 32'shFFFFFDAC;
            8'd131: data = 32'shFFFFFDF7;
            8'd132: data = 32'shFFFFFAEF;
            8'd133: data = 32'shFFFFFC49;
            8'd134: data = 32'sh000006A3;
            8'd135: data = 32'shFFFFFEF4;
            8'd136: data = 32'shFFFFFF5E;
            8'd137: data = 32'sh000003F4;
            8'd138: data = 32'shFFFFFF8D;
            8'd139: data = 32'shFFFFFE64;
            8'd140: data = 32'sh00000757;
            8'd141: data = 32'sh0000029F;
            8'd142: data = 32'shFFFFFD12;
            8'd143: data = 32'shFFFFFE88;

            // ds_blocks.2.pointwise (offset 144, 24 values)
            8'd144: data = 32'sh0000013D;
            8'd145: data = 32'shFFFFFEA5;
            8'd146: data = 32'shFFFFF964;
            8'd147: data = 32'sh00000158;
            8'd148: data = 32'shFFFFF744;
            8'd149: data = 32'shFFFFFC73;
            8'd150: data = 32'sh00000393;
            8'd151: data = 32'sh00000177;
            8'd152: data = 32'shFFFFFCBC;
            8'd153: data = 32'sh00000147;
            8'd154: data = 32'sh00000AF7;
            8'd155: data = 32'sh000003BE;
            8'd156: data = 32'sh0000009D;
            8'd157: data = 32'shFFFFF312;
            8'd158: data = 32'sh00000173;
            8'd159: data = 32'sh00000202;
            8'd160: data = 32'sh0000007C;
            8'd161: data = 32'shFFFFF585;
            8'd162: data = 32'sh000005E4;
            8'd163: data = 32'shFFFFFA67;
            8'd164: data = 32'sh0000018A;
            8'd165: data = 32'shFFFFFA56;
            8'd166: data = 32'shFFFFFBF6;
            8'd167: data = 32'shFFFFF9F6;

            // ds_blocks.3.depthwise (offset 168, 24 values)
            8'd168: data = 32'shFFFFFC59;
            8'd169: data = 32'sh0000030F;
            8'd170: data = 32'shFFFFFC8B;
            8'd171: data = 32'shFFFFFCB7;
            8'd172: data = 32'sh0000035A;
            8'd173: data = 32'shFFFFFE55;
            8'd174: data = 32'shFFFFFF15;
            8'd175: data = 32'shFFFFFFBD;
            8'd176: data = 32'shFFFFFAFF;
            8'd177: data = 32'shFFFFFCBA;
            8'd178: data = 32'sh00000149;
            8'd179: data = 32'sh00000264;
            8'd180: data = 32'shFFFFFEC9;
            8'd181: data = 32'shFFFFFD19;
            8'd182: data = 32'shFFFFF9C9;
            8'd183: data = 32'shFFFFFE16;
            8'd184: data = 32'shFFFFFD57;
            8'd185: data = 32'sh000001A8;
            8'd186: data = 32'sh000002C7;
            8'd187: data = 32'shFFFFFBF0;
            8'd188: data = 32'sh000002DA;
            8'd189: data = 32'shFFFFFE1F;
            8'd190: data = 32'shFFFFFF01;
            8'd191: data = 32'shFFFFFBB1;

            // ds_blocks.3.pointwise (offset 192, 24 values)
            8'd192: data = 32'shFFFFFFB8;
            8'd193: data = 32'sh00000004;
            8'd194: data = 32'shFFFFFF7E;
            8'd195: data = 32'shFFFFFFCE;
            8'd196: data = 32'shFFFFFF29;
            8'd197: data = 32'shFFFFFFB7;
            8'd198: data = 32'sh000000CB;
            8'd199: data = 32'sh0000016D;
            8'd200: data = 32'shFFFFFF88;
            8'd201: data = 32'shFFFFFF48;
            8'd202: data = 32'sh00000009;
            8'd203: data = 32'shFFFFFFC9;
            8'd204: data = 32'sh00000210;
            8'd205: data = 32'sh0000004A;
            8'd206: data = 32'shFFFFFFD5;
            8'd207: data = 32'sh000000B1;
            8'd208: data = 32'shFFFFFF62;
            8'd209: data = 32'shFFFFFFDA;
            8'd210: data = 32'shFFFFFF62;
            8'd211: data = 32'shFFFFFFB5;
            8'd212: data = 32'shFFFFFF90;
            8'd213: data = 32'sh00000047;
            8'd214: data = 32'sh00000092;
            8'd215: data = 32'shFFFFFF78;

            // classifier (offset 216, 7 values)
            8'd216: data = 32'shFFFFFFFF;
            8'd217: data = 32'sh00000012;
            8'd218: data = 32'shFFFFFFC0;
            8'd219: data = 32'shFFFFFFA8;
            8'd220: data = 32'sh00000085;
            8'd221: data = 32'sh00000001;
            8'd222: data = 32'shFFFFFFF7;

            default: data = 32'sh00000000;
        endcase
    end

endmodule