// SPDX-License-Identifier: Apache-2.0

// Authour: Claude

`timescale 1ns/1ps
`ifndef GF180MCU_OCD_IP_SRAM_MODELS_V
`define GF180MCU_OCD_IP_SRAM_MODELS_V

module gf180mcu_ocd_ip_sram__sram256x8m8wm1 (
    CLK,
    CEN,
    GWEN,
    WEN,
    A,
    D,
    Q
`ifdef USE_POWER_PINS
    , VDD, VSS
`endif
);
    input         CLK;
    input         CEN;
    input         GWEN;
    input  [7:0]  WEN;
    input  [7:0]  A;
    input  [7:0]  D;
    output [7:0]  Q;
`ifdef USE_POWER_PINS
    inout         VDD;
    inout         VSS;
`endif

    reg [7:0] mem [0:255];
    reg [7:0] q_r;
    integer   k;
    initial begin
        for (k = 0; k < 256; k = k + 1) mem[k] = 8'h00;
        q_r = 8'h00;
    end
    always @(posedge CLK) begin
        if (CEN === 1'b0) begin
            if (GWEN === 1'b0) begin
                if (WEN[0] === 1'b0) mem[A][0] <= D[0];
                if (WEN[1] === 1'b0) mem[A][1] <= D[1];
                if (WEN[2] === 1'b0) mem[A][2] <= D[2];
                if (WEN[3] === 1'b0) mem[A][3] <= D[3];
                if (WEN[4] === 1'b0) mem[A][4] <= D[4];
                if (WEN[5] === 1'b0) mem[A][5] <= D[5];
                if (WEN[6] === 1'b0) mem[A][6] <= D[6];
                if (WEN[7] === 1'b0) mem[A][7] <= D[7];
            end else begin
                q_r <= mem[A];
            end
        end
    end
    assign Q = q_r;
`ifndef FUNCTIONAL
    specify
        (CLK *> Q) = (0:0:0, 0:0:0);
    endspecify
`endif
endmodule

module gf180mcu_ocd_ip_sram__sram512x8m8wm1 (
    CLK,
    CEN,
    GWEN,
    WEN,
    A,
    D,
    Q
`ifdef USE_POWER_PINS
    , VDD, VSS
`endif
);
    input         CLK;
    input         CEN;
    input         GWEN;
    input  [7:0]  WEN;
    input  [8:0]  A;
    input  [7:0]  D;
    output [7:0]  Q;
`ifdef USE_POWER_PINS
    inout         VDD;
    inout         VSS;
`endif

    reg [7:0] mem [0:511];
    reg [7:0] q_r;
    integer   k;
    initial begin
        for (k = 0; k < 512; k = k + 1) mem[k] = 8'h00;
        q_r = 8'h00;
    end
    always @(posedge CLK) begin
        if (CEN === 1'b0) begin
            if (GWEN === 1'b0) begin
                if (WEN[0] === 1'b0) mem[A][0] <= D[0];
                if (WEN[1] === 1'b0) mem[A][1] <= D[1];
                if (WEN[2] === 1'b0) mem[A][2] <= D[2];
                if (WEN[3] === 1'b0) mem[A][3] <= D[3];
                if (WEN[4] === 1'b0) mem[A][4] <= D[4];
                if (WEN[5] === 1'b0) mem[A][5] <= D[5];
                if (WEN[6] === 1'b0) mem[A][6] <= D[6];
                if (WEN[7] === 1'b0) mem[A][7] <= D[7];
            end else begin
                q_r <= mem[A];
            end
        end
    end
    assign Q = q_r;
`ifndef FUNCTIONAL
    specify
        (CLK *> Q) = (0:0:0, 0:0:0);
    endspecify
`endif
endmodule

module gf180mcu_ocd_ip_sram__sram1024x8m8wm1 (
    CLK,
    CEN,
    GWEN,
    WEN,
    A,
    D,
    Q
`ifdef USE_POWER_PINS
    , VDD, VSS
`endif
);
    input         CLK;
    input         CEN;
    input         GWEN;
    input  [7:0]  WEN;
    input  [9:0]  A;
    input  [7:0]  D;
    output [7:0]  Q;
`ifdef USE_POWER_PINS
    inout         VDD;
    inout         VSS;
`endif

    reg [7:0] mem [0:1023];
    reg [7:0] q_r;
    integer   k;
    initial begin
        for (k = 0; k < 1024; k = k + 1) mem[k] = 8'h00;
        q_r = 8'h00;
    end
    always @(posedge CLK) begin
        if (CEN === 1'b0) begin
            if (GWEN === 1'b0) begin
                if (WEN[0] === 1'b0) mem[A][0] <= D[0];
                if (WEN[1] === 1'b0) mem[A][1] <= D[1];
                if (WEN[2] === 1'b0) mem[A][2] <= D[2];
                if (WEN[3] === 1'b0) mem[A][3] <= D[3];
                if (WEN[4] === 1'b0) mem[A][4] <= D[4];
                if (WEN[5] === 1'b0) mem[A][5] <= D[5];
                if (WEN[6] === 1'b0) mem[A][6] <= D[6];
                if (WEN[7] === 1'b0) mem[A][7] <= D[7];
            end else begin
                q_r <= mem[A];
            end
        end
    end
    assign Q = q_r;
`ifndef FUNCTIONAL
    specify
        (CLK *> Q) = (0:0:0, 0:0:0);
    endspecify
`endif
endmodule

`endif