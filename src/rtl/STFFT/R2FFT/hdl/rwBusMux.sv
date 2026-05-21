/*
 * rwBusMux not USED
 *
 * combined read/write bus mux for ONE
 * single-port RAM, one source per cycle selected by the (now retired) main
 * FSM state.
 *   ST_INPUT_STREAM -> input stream writes
 *   ST_RUN_FFT      -> butterfly read or write
 *   ST_DONE         -> DMA reads
 *   otherwise       -> idle
 *
 */
module rwBusMux
  #(parameter FFT_N  = 10,
    parameter FFT_DW = 16)
   (
    input  wire                  active,
    input  wire [2:0]            state,
 
    // input stream
    input  wire                  istream_act,
    input  wire [FFT_N-1:0]      istream_a,
    input  wire [FFT_DW*2-1:0]   istream_dw,
 
    // DMA
    input  wire                  dma_act,
    input  wire [FFT_N-1:0]      dma_a,
 
    // butterfly
    input  wire                  bf_act,
    input  wire                  bf_we,
    input  wire [FFT_N-1:0]      bf_a,
    input  wire [FFT_DW*2-1:0]   bf_dw,
 
    // single-port RAM
    output reg                   ram_act,
    output reg                   ram_we,
    output reg [FFT_N-1:0]       ram_a,
    output reg [FFT_DW*2-1:0]    ram_dw
    );
 
   // must match status_t encoding in R2FFT
   localparam ST_INPUT_STREAM = 3'd1;
   localparam ST_RUN_FFT      = 3'd3;
   localparam ST_DONE         = 3'd4;
 
   always_comb begin
      ram_act = 1'b0;
      ram_we  = 1'b0;
      ram_a   = '0;
      ram_dw  = '0;
 
      if (active) begin
         case (state)
           ST_INPUT_STREAM: begin
              ram_act = istream_act;
              ram_we  = 1'b1;
              ram_a   = istream_a;
              ram_dw  = istream_dw;
           end
           ST_RUN_FFT: begin
              ram_act = bf_act;
              ram_we  = bf_we;
              ram_a   = bf_a;
              ram_dw  = bf_dw;
           end
           ST_DONE: begin
              ram_act = dma_act;
              ram_we  = 1'b0;
              ram_a   = dma_a;
              ram_dw  = '0;
           end
           default: ; // idle
         endcase
      end
   end
 
endmodule // rwBusMux
