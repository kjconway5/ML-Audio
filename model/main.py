import numpy as np
from pathlib import Path

from pipeline import Preprocessing
from utils import (
    load_wav, save_wav, 
    generate_chirp, visualize_pipeline, 
    generate_rtl_test_vectors
)


def process_audio_file(input_wav, output_dir='output', hpf_order_low=2, lpf_order_low=4,
        hpf_order_high=8, lpf_order_high=8, visualize=True, save_outputs=True, generate_test_vectors_flag=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # load audio
    audio, sample_rate = load_wav(input_wav)
    
    # create preprocessors
    preprocessor_low = Preprocessing(
        sample_rate=sample_rate,
        hpf_order=hpf_order_low,
        lpf_order=lpf_order_low
    )
    print(f"Filter config: {preprocessor_low.filter.get_config()}")
    
    print("\nHigh-Order Configuration")
    preprocessor_high = Preprocessing(
        sample_rate=sample_rate,
        hpf_order=hpf_order_high,
        lpf_order=lpf_order_high
    )
    print(f"Filter config: {preprocessor_high.filter.get_config()}")
    
    features_low, filtered_low, time_low = preprocessor_low.process(audio)
    features_high, filtered_high, time_high = preprocessor_high.process(audio)
    
    results = {
        'original_audio': audio,
        'filtered_low': filtered_low,
        'filtered_high': filtered_high,
        'features_low': features_low,
        'features_high': features_high,
        'time_frames_low': time_low,
        'time_frames_high': time_high,
        'sample_rate': sample_rate,
        'config_low': preprocessor_low.get_config(),
        'config_high': preprocessor_high.get_config()
    }
    
    if save_outputs:
        print("\nSaving Outputs")
        save_wav(output_dir / 'filtered_low.wav', filtered_low, sample_rate)
        save_wav(output_dir / 'filtered_high.wav', filtered_high, sample_rate)
        np.save(output_dir / 'features_low.npy', features_low)
        np.save(output_dir / 'features_high.npy', features_high)
        print(f"  Saved to {output_dir}/")
    
    if generate_test_vectors_flag:
        print("\nGenerating RTL Test Vectors")
        generate_rtl_test_vectors(audio, output_dir, 'input_original')
        generate_rtl_test_vectors(filtered_low, output_dir, 'output_low')
        generate_rtl_test_vectors(filtered_high, output_dir, 'output_high')
    
    if visualize:
        print("\nGenerating Visualization")
        fig = visualize_pipeline(
            audio, filtered_low, filtered_high,
            features_low, features_high,
            sample_rate,
            preprocessor_low.get_config(),
            preprocessor_high.get_config()
        )
        fig.savefig(output_dir / 'pipeline_verification.png', dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_dir}/pipeline_verification.png")
        import matplotlib.pyplot as plt
        plt.show()
    
    return results


def main():
    print("\nChoose test signal:")
    print("1. Chirp")
    print("2. Load existing WAV file")
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        print("\nGenerating chirp...")
        audio = generate_chirp(duration=3, sample_rate=16000)
        save_wav('chirp_test.wav', audio, 16000)
        input_wav = 'chirp_test.wav'
    else:
        input_wav = input("Enter path to WAV file: ").strip()
    
    results = process_audio_file(
        input_wav=input_wav,
        output_dir='output',
        hpf_order_low=2,
        lpf_order_low=4,
        hpf_order_high=8,
        lpf_order_high=8,
        visualize=True,
        save_outputs=True,
        generate_test_vectors_flag=True
    )


if __name__ == "__main__":
    main()