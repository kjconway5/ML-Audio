module frame 
    #(parameter FRAME_LEN = 160, //160 = 20 ms frame at 8kHz 
      parameter HOP_LEN = 160,  
      parameter WIDTH = $clog2(FRAME_LEN)) 
    input logic clk, 
    input logic reset, 
    input logic sample_valid,
    input logic sample_ready, 
    output logic take_sample, 
    output logic frame_start,
    output logic frame_end,
    output logic hop_start,
    output logic hop_end
    );

    //Framing to output frame_start and frame_end based on desired frame length
    //FRAME_LEN = Fs * (ms/1000) , Fs = sample rate, ms is frame length usually (10-25) 
    //This module computes the frame size, multiple frames are analyzed for around a .5-1 second window 
    //Rate of count incrementation must be based on clk, sample rate and desired frame length
    

    logic [WIDTH-1:0] frame_cnt;

    assign take_sample = sample_valid && sample_ready; 
    assign frame_start = (take_sample && (frame_cnt == '0));
    assign frame_end = (take_sample && (frame_cnt == FRAME_LEN-1'b1)); 

    always_ff @(posedge clk) begin
        if (reset) begin 
            frame_cnt <= '0; 
        //    frame_start <= '0;                
        //    frame_end <= '0;                  //Probably unnecessary 
        end else if (take_sample) begin 
            if (frame_cnt == FRAME_LEN-1'b1) begin 
                frame_cnt <= '0; 
            end else begin 
                frame_cnt <= frame_cnt + 1'b1; 
            end 
        end 
    end

    //Hop length is used for overlap between frames. More overlap less chance of audio loss 
    //   (if event occurs between frames) 
    //Hop = Frame means no overlap so not necessary right now
   

    

    







