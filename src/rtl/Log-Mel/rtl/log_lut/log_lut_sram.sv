module log_lut_sram #(
    parameter int LUT_DEPTH = 64,
    parameter int DATA_W    = 16,
    parameter int ADDR_W    = 6     
)(
    input  wire [0:0] clk_i,

    //FLASH 
    input  wire  [0:0] flash_write_enable_i,
    input  wire [ADDR_W-1:0] flash_addr_i,    
    input  wire [DATA_W-1:0] flash_write_data_i,  

    //Runtime Read
    input  wire [ADDR_W-1:0] rd_addr_i,
    output wire [DATA_W-1:0] rd_data_o
);
    //Flash write vs run time read
    wire [ADDR_W-1:0] sram_addr = flash_write_enable_i ? flash_addr_i : rd_addr_i;
    wire [0:0] gwen  = flash_write_enable_i ? 1'b0 : 1'b1;  // 0=write, 1=read
    wire [0:0] cen   = 1'b0;   

    wire [7:0] wen = flash_write_enable_i ? 8'h00 : 8'hFF;

    wire [7:0] q_lo, q_hi;

    //Low byte
    gf180mcu_fd_ip_sram__sram64x8m8wm1 u_lut_lo (
        .CLK  (clk_i),
        .CEN  (cen),
        .GWEN (gwen),
        .WEN  (wen),
        .A    (sram_addr),
        .D    (flash_write_data_i[7:0]),
        .Q    (q_lo),
        .VDD  (1'b1),
        .VSS  (1'b0)
    );

    //High byte
    gf180mcu_fd_ip_sram__sram64x8m8wm1 u_lut_hi (
        .CLK  (clk_i),
        .CEN  (cen),
        .GWEN (gwen),
        .WEN  (wen),
        .A    (sram_addr),
        .D    (flash_write_data_i[15:8]),
        .Q    (q_hi),
        .VDD  (1'b1),
        .VSS  (1'b0)
    );

    assign rd_data_o = {q_hi, q_lo};

endmodule