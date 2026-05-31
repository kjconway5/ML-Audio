// Vendored copy of AvalonSemiconductors/gf180mcu_as_sc_mcu7t3v3 verilog model.
// Authors: Claude, based on PDK models by GF and Avalon. Rewritten for GLS by Claude.

`timescale 1ns/1ps

module gf180mcu_as_sc_mcu7t3v3__dfxtp_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input CLK,
	input D,
	output Q
);

reg state;
always @(posedge CLK) state <= D;
assign Q = state;


`ifndef FUNCTIONAL
specify
	(posedge CLK => (Q:D)) = (0:0:0, 0:0:0);
	$setup(posedge D, posedge CLK, 0:0:0);
	$setup(negedge D, posedge CLK, 0:0:0);
	$hold(posedge CLK, posedge D, 0:0:0);
	$hold(posedge CLK, negedge D, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__dfxtn_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input CLK,
	input D,
	output Q
);

reg state;
always @(negedge CLK) state <= D;
assign Q = state;


`ifndef FUNCTIONAL
specify
	(negedge CLK => (Q:D)) = (0:0:0, 0:0:0);
	$setup(posedge D, negedge CLK, 0:0:0);
	$setup(negedge D, negedge CLK, 0:0:0);
	$hold(negedge CLK, posedge D, 0:0:0);
	$hold(negedge CLK, negedge D, 0:0:0);
endspecify
`endif

endmodule

// dfsrtp_2: D flip-flop, async active-low reset (RN) and set (SN).
// PDK shipped a broken model (output Q declared as wire but assigned in
// always block; `state` was never driven). Rewritten as a conventional
// async-set/reset DFF — required for GLS to elaborate.
module gf180mcu_as_sc_mcu7t3v3__dfsrtp_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input CLK,
	input D,
	input RN,
	input SN,
	output Q
);

reg q_r;
always @(posedge CLK or negedge RN or negedge SN) begin
	if (!RN)      q_r <= 1'b0;
	else if (!SN) q_r <= 1'b1;
	else          q_r <= D;
end
assign Q = q_r;


`ifndef FUNCTIONAL
specify
	(posedge CLK => (Q:D)) = (0:0:0, 0:0:0);
	$setup(posedge D, posedge CLK, 0:0:0);
	$setup(negedge D, posedge CLK, 0:0:0);
	$hold(posedge CLK, posedge D, 0:0:0);
	$hold(posedge CLK, negedge D, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__buff_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__buff_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__buff_8(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__buff_12(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__clkbuff_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__clkbuff_8(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__clkbuff_12(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__inv_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = !A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__inv_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = !A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__inv_6(
	input VPW,
	input VNW,
	input VDD,
	input VSS,
	
	input A,
	output Y
);

assign Y = !A;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__invz_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input EN,
	output Y
);

assign Y = (EN == 1'b1) ? ~A : 1'bz;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge EN => (Y:EN)) = (0:0:0, 0:0:0);
	(negedge EN => (Y:EN)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nand2_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = !(A & B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nand2_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = !(A & B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nand2b_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A | (!B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nand2b_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A | (!B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nand3_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = !(A & B & C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nand4_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = !(A & B & C & D);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nor2_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = !(A | B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nor2_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = !(A | B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nor2b_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A & (!B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nor2b_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A & (!B);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__nor3_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = !(A | B | C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__and2_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A & B;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__and2_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A & B;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__or2_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A | B;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__or2_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A | B;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__xnor2_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A == B;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__xnor2_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	output Y
);

assign Y = A == B;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__maj3_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = (A & B) | (A & C) | (B & C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__maj3_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = (A & B) | (A & C) | (B & C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi21_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = ~((A & B) | C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi21_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = ~((A & B) | C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi21b_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = ~(((~A) & B) | C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi21b_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = ~(((~A) & B) | C);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao21_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = (A & B) | C;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao21_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = (A & B) | C;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao21b_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = ((~A) & B) | C;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao21b_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	output Y
);

assign Y = ((~A) & B) | C;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi22_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = ~((A & B) | (C & D));


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi22_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = ~((A & B) | (C & D));


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao22_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = (A & B) | (C & D);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao22_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = (A & B) | (C & D);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi31_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = ~((A & B & C) | D);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__aoi31_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = ~((A & B & C) | D);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao31_2 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = (A & B & C) | D;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__ao31_4 (
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input C,
	input D,
	output Y
);

assign Y = (A & B & C) | D;


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge C => (Y:C)) = (0:0:0, 0:0:0);
	(negedge C => (Y:C)) = (0:0:0, 0:0:0);
	(posedge D => (Y:D)) = (0:0:0, 0:0:0);
	(negedge D => (Y:D)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__mux2_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input S,
	output Y
);

assign Y = (S&B) | ((!S)&A);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge S => (Y:S)) = (0:0:0, 0:0:0);
	(negedge S => (Y:S)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__mux2_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input A,
	input B,
	input S,
	output Y
);

assign Y = (S&B) | ((!S)&A);


`ifndef FUNCTIONAL
specify
	(posedge A => (Y:A)) = (0:0:0, 0:0:0);
	(negedge A => (Y:A)) = (0:0:0, 0:0:0);
	(posedge B => (Y:B)) = (0:0:0, 0:0:0);
	(negedge B => (Y:B)) = (0:0:0, 0:0:0);
	(posedge S => (Y:S)) = (0:0:0, 0:0:0);
	(negedge S => (Y:S)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

module gf180mcu_as_sc_mcu7t3v3__tap_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fill_1(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fill_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fill_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fill_8(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fillcap_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fillcap_8(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__fillcap_16(
	input VPW,
	input VNW,
	input VDD,
	input VSS
);

endmodule

module gf180mcu_as_sc_mcu7t3v3__tieh_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	output ONE
);

assign ONE = 1'b1;

endmodule

module gf180mcu_as_sc_mcu7t3v3__tiel_4(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	output ZERO
);

assign ZERO = 1'b0;

endmodule

module gf180mcu_as_sc_mcu7t3v3__diode_2(
	input VPW,
	input VNW,
	input VDD,
	input VSS,

	input DIODE
);

endmodule