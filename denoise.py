#!/usr/bin/env python3
"""
Audio Denoising Script
This script takes khabib.mp3, applies noise reduction, and saves the result as khabib_clean.mp3
"""

import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import os
import sys


def denoise_audio(input_file, output_file):
    """
    Denoise an audio file using spectral subtraction

    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to output denoised audio file
    """

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return False

    try:
        print(f"Loading audio file: {input_file}")

        # Load the audio file
        # librosa loads audio as floating point time series
        audio_data, sample_rate = librosa.load(input_file, sr=None)

        print(f"Audio loaded successfully!")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"Audio shape: {audio_data.shape}")

        # Apply noise reduction
        print("Applying noise reduction...")

        # Estimate noise from the first 0.5 seconds of audio
        # This assumes the first part contains mostly noise
        noise_sample_length = min(int(0.5 * sample_rate), len(audio_data) // 4)

        # Perform noise reduction using spectral subtraction
        denoised_audio = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            stationary=True,  # Assume stationary noise
            prop_decrease=0.8,  # Proportion to decrease noise by
        )

        # Alternative method: using non-stationary noise reduction
        # This can work better for varying background noise
        # denoised_audio = nr.reduce_noise(
        #     y=audio_data,
        #     sr=sample_rate,
        #     stationary=False,
        #     prop_decrease=0.8
        # )

        print("Noise reduction completed!")

        # Save the denoised audio
        print(f"Saving denoised audio to: {output_file}")
        sf.write(output_file, denoised_audio, sample_rate)

        print(f"Denoised audio saved successfully!")
        print(f"Original file size: {os.path.getsize(input_file)} bytes")
        print(f"Denoised file size: {os.path.getsize(output_file)} bytes")

        return True

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return False


def main():
    """Main function to run the denoising process"""

    input_file = "khabib.mp3"
    output_file = "khabib_clean.mp3"

    print("=" * 50)
    print("Audio Denoising Script")
    print("=" * 50)

    # Check if required libraries are available
    try:
        import librosa
        import soundfile
        import noisereduce

        print("All required libraries are available!")
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install required packages:")
        print("pip install librosa soundfile noisereduce")
        sys.exit(1)

    # Perform denoising
    success = denoise_audio(input_file, output_file)

    if success:
        print("\n✅ Denoising completed successfully!")
        print(f"Original file: {input_file}")
        print(f"Denoised file: {output_file}")
    else:
        print("\n❌ Denoising failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
