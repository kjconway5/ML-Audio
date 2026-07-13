// SPDX-License-Identifier: Apache-2.0
//
// Auhtor: Claude

`timescale 1ns/1ps
`ifndef GF180MCU_FD_IO_V_SIM
`define GF180MCU_FD_IO_V_SIM

module gf180mcu_fd_io__dvdd (DVDD, DVSS, VSS);
    inout DVDD, DVSS, VSS;
endmodule

module gf180mcu_fd_io__dvss (DVDD, DVSS, VDD);
    inout DVDD, DVSS, VDD;
endmodule

module gf180mcu_fd_io__in_c (PU, PD, PAD, Y, DVDD, DVSS, VDD, VSS);
    input  PU, PD;
    inout  PAD;
    output Y;
    inout  DVDD, DVSS, VDD, VSS;
    assign Y = PAD;
`ifndef FUNCTIONAL
    specify
        (PAD => Y) = (0:0:0, 0:0:0);
    endspecify
`endif
endmodule

// Schmitt-trigger input pad
module gf180mcu_fd_io__in_s (PU, PD, PAD, Y, DVDD, DVSS, VDD, VSS);
    input  PU, PD;
    inout  PAD;
    output Y;
    inout  DVDD, DVSS, VDD, VSS;
    assign Y = PAD;
`ifndef FUNCTIONAL
    specify
        (PAD => Y) = (0:0:0, 0:0:0);
    endspecify
`endif
endmodule

// Bidirectional 24mA pad
module gf180mcu_fd_io__bi_24t (CS, SL, IE, OE, PU, PD, A, PAD, Y, DVDD, DVSS, VDD, VSS);
    input  CS, SL, IE, OE, PU, PD, A;
    inout  PAD;
    output Y;
    inout  DVDD, DVSS, VDD, VSS;
    assign PAD = OE ? A : 1'bz;
    assign Y   = IE ? PAD : 1'b0;
`ifndef FUNCTIONAL
    specify
        (PAD => Y)  = (0:0:0, 0:0:0);
        (A   => PAD) = (0:0:0, 0:0:0);
        (OE  => PAD) = (0:0:0, 0:0:0);
        (IE  => Y)   = (0:0:0, 0:0:0);
    endspecify
`endif
endmodule

// Analog signal pad — pure passthrough
module gf180mcu_fd_io__asig_5p0 (ASIG5V, DVDD, DVSS, VDD, VSS);
    inout ASIG5V;
    inout DVDD, DVSS, VDD, VSS;
endmodule

`endif

`ifndef GF180MCU_FD_IO_FILLER_CORNER_MISSING_MODELS
`define GF180MCU_FD_IO_FILLER_CORNER_MISSING_MODELS

module gf180mcu_fd_io__cor (
    inout DVDD,
    inout DVSS,
    inout VDD,
    inout VSS
);
endmodule

module gf180mcu_fd_io__fill10 (
    inout DVDD,
    inout DVSS,
    inout VDD,
    inout VSS
);
endmodule

module gf180mcu_fd_io__fill5 (
    inout DVDD,
    inout DVSS,
    inout VDD,
    inout VSS
);
endmodule

module gf180mcu_fd_io__fill1 (
    inout DVDD,
    inout DVSS,
    inout VDD,
    inout VSS
);
endmodule

module gf180mcu_fd_io__fillnc (
    inout DVDD,
    inout DVSS,
    inout VDD,
    inout VSS
);
endmodule

`endif