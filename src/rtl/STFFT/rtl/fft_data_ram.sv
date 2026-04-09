module fft_data_ram (
    input  wire        clk,
    input  wire        ract,
    input  wire        wact,
    input  wire [6:0]  addr,
    input  wire [31:0] wdata,
    output wire [31:0] rdata
);
    wire cen  = ~(ract | wact);
    wire gwen = ~wact;
    wire [7:0] wen = {8{~wact}};
    wire [7:0] q0, q1, q2, q3;

    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b0 (.CLK(clk),.CEN(cen),.GWEN(gwen),.WEN(wen),.A(addr),.D(wdata[7:0]),  .Q(q0));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b1 (.CLK(clk),.CEN(cen),.GWEN(gwen),.WEN(wen),.A(addr),.D(wdata[15:8]), .Q(q1));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b2 (.CLK(clk),.CEN(cen),.GWEN(gwen),.WEN(wen),.A(addr),.D(wdata[23:16]),.Q(q2));
    gf180mcu_fd_ip_sram__sram128x8m8wm1 u_b3 (.CLK(clk),.CEN(cen),.GWEN(gwen),.WEN(wen),.A(addr),.D(wdata[31:24]),.Q(q3));

    assign rdata = {q3, q2, q1, q0};
endmodule