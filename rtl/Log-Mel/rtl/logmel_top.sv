module logmel_top #(
    parameter FFT_SIZE = 256,
    parameter MEL_BINS = 40,
    parameter IW = 18, // input width from STFT
    parameter OW = 16  // output width to CNN
)
(
    input  [35:0] fft_data,  // directly from o_fft_result
    input         fft_valid, // from o_fft_sync (or derived)
    input         fft_sync,  // frame start pulse

    output [OW-1:0] mel_out,
    output          mel_valid,
    output          mel_frame_done
)