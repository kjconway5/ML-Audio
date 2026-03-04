pipeline_top #(

) (

);

    // TODO: Add ports as needed
    // order:
    // 1. PDM Decimator (waiting on these for later)
    // 2. FIR Filter (waiting on these for later)
    // 3. FIFO
    // 4. STFFT
    // 5. LogMel Top
    // 6. Output to ML

    // Parameters 
    // 

    // Inputs
    //

    // Outputs 
    // 


    fifo_1r1w #(
        width_p = 8,
        depth_log2_p = 8
    ) (
        .clk_i(),
        .reset_i(),
        .data_i(),
        .valid_i(),
        .ready_i(),
        .ready_o(),
        .valid_o(),
        .data_o()
    );

    stfft #(
        .IW(),
        .OW(),
        .FFT_SIZE()
    ) fft (
        .i_clk()
        .i_reset()
        .i_ce()
        .i_sample()
        .o_fft_result()
        .o_fft_sync()
    );

    logmel_top #(
        .IW(),
        .SHIFT(),
        .N_MELS(),
        .N_BINS(),
        .MAX_COEFFS(),
        .POWER_W(),
        .WEIGHT_W(),
        .ACCUM_W (),
        .LOG_OUT_W(),
        .OUT_W()
    ) logmel (
        .clk_i(),
        .reset_i(),

        // from STFT
        .re_il(),
        .im_il(),
        .fft_valid_il(),
        .fft_sync_il(),

        // to CNN
        .cnn_data_ol(),
        .cnn_valid_ol(),
        .cnn_ready_il()
    );


