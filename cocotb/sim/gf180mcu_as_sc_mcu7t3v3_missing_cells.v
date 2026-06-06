// SPDX-License-Identifier: Apache-2.0
//
// Author: Marco Frank/Claude

// Behavioural stubs for standard cells that are missing from the shipped
// gf180mcu_as_sc_mcu7t3v3.v (the gf180mcuD PDK ships an incomplete library
// file for the 3.3V AS std-cell set).  These cells ARE present in the .lib
// timing files and ARE used by the synthesised netlist; without these
// stubs Icarus errors with "Unknown module type".
//
// Boolean functions taken verbatim from
//   gf180mcu_as_sc_mcu7t3v3__tt_025C_3v30.lib:
//     ao211 :  Y = (A & B) | C | D
//     aoi211: Y = !((A & B) | C | D)
//     oai211: Y = !((A | B) & C & D)
//     xor2  : Y = (A & !B) | (B & !A)     // = A ^ B
//
// Pin order matches the netlist instantiations (and the convention used by
// the other ao/aoi/oai cells in gf180mcu_as_sc_mcu7t3v3.v): power pins
// VPW/VNW/VDD/VSS first, then signal inputs A,B[,C,D], output Y.

`timescale 1ns/1ps

module gf180mcu_as_sc_mcu7t3v3__ao211_2 (
    input  VPW,
    input  VNW,
    input  VDD,
    input  VSS,
    input  A,
    input  B,
    input  C,
    input  D,
    output Y
);
    assign Y = (A & B) | C | D;
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

module gf180mcu_as_sc_mcu7t3v3__aoi211_2 (
    input  VPW,
    input  VNW,
    input  VDD,
    input  VSS,
    input  A,
    input  B,
    input  C,
    input  D,
    output Y
);
    assign Y = ~((A & B) | C | D);
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

module gf180mcu_as_sc_mcu7t3v3__oai211_2 (
    input  VPW,
    input  VNW,
    input  VDD,
    input  VSS,
    input  A,
    input  B,
    input  C,
    input  D,
    output Y
);
    assign Y = ~((A | B) & C & D);
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

// _4 drive-strength variants of ao211 and aoi211. Same boolean function as
// the _2 cells; higher drive strength is a layout-only difference that does
// not affect functional simulation. Required because the post-PnR netlist
// for the current run (e.g. RUN_2026-05-15_11-55-49) uses these strengths
// after resizer optimization, even though the post-synth netlist only used _2.
module gf180mcu_as_sc_mcu7t3v3__ao211_4 (
    input  VPW,
    input  VNW,
    input  VDD,
    input  VSS,
    input  A,
    input  B,
    input  C,
    input  D,
    output Y
);
    assign Y = (A & B) | C | D;
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

module gf180mcu_as_sc_mcu7t3v3__aoi211_4 (
    input  VPW,
    input  VNW,
    input  VDD,
    input  VSS,
    input  A,
    input  B,
    input  C,
    input  D,
    output Y
);
    assign Y = ~((A & B) | C | D);
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

`ifndef GF180MCU_AS_SC_MCU7T3V3_DFXTP_4_MISSING_MODEL
`define GF180MCU_AS_SC_MCU7T3V3_DFXTP_4_MISSING_MODEL

module gf180mcu_as_sc_mcu7t3v3__dfxtp_4 (
`ifdef USE_POWER_PINS
    inout VDD,
    inout VNW,
    inout VPW,
    inout VSS,
`endif
    input CLK,
    input D,
    output reg Q
);
    always @(posedge CLK) begin
        Q <= D;
    end

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

`endif

`ifndef GF180MCU_AS_SC_MCU7T3V3_XOR2_2_MISSING_MODEL
`define GF180MCU_AS_SC_MCU7T3V3_XOR2_2_MISSING_MODEL

module gf180mcu_as_sc_mcu7t3v3__xor2_2 (
`ifdef USE_POWER_PINS
    inout VDD,
    inout VNW,
    inout VPW,
    inout VSS,
`endif
    input A,
    input B,
    output Y
);
    assign Y = A ^ B;

`ifndef FUNCTIONAL
specify
      (posedge A => (Y:A)) = (0:0:0, 0:0:0);
      (negedge A => (Y:A)) = (0:0:0, 0:0:0);
      (posedge B => (Y:B)) = (0:0:0, 0:0:0);
      (negedge B => (Y:B)) = (0:0:0, 0:0:0);
endspecify
`endif

endmodule

`endif

`ifndef GF180MCU_AS_SC_MCU7T3V3_DLYBUFF_2_MISSING_MODEL
`define GF180MCU_AS_SC_MCU7T3V3_DLYBUFF_2_MISSING_MODEL

module gf180mcu_as_sc_mcu7t3v3__dlybuff_2 (
`ifdef USE_POWER_PINS
    inout VDD,
    inout VNW,
    inout VPW,
    inout VSS,
`endif
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

`endif

`ifndef GF180MCU_AS_SC_MCU7T3V3_BUFF_16_MISSING_MODEL
`define GF180MCU_AS_SC_MCU7T3V3_BUFF_16_MISSING_MODEL

module gf180mcu_as_sc_mcu7t3v3__buff_16 (
`ifdef USE_POWER_PINS
    inout VDD,
    inout VNW,
    inout VPW,
    inout VSS,
`endif
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

`endif