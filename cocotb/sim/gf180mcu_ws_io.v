// SPDX-License-Identifier: Apache-2.0

// Authour: Claude

`timescale 1ns/1ps
`ifndef GF180MCU_WS_IO_V_SIM
`define GF180MCU_WS_IO_V_SIM

module gf180mcu_ws_io__dvdd (DVDD, DVSS, VSS);
    inout DVDD, DVSS, VSS;
endmodule

module gf180mcu_ws_io__dvss (DVDD, DVSS, VDD);
    inout DVDD, DVSS, VDD;
endmodule

`endif