import os
import ffmpeg
from pathlib import Path


def extract_audio_to_mp3(video_path, output_path=None, overwrite=False):
    """
    Extract audio from a video file and save it as MP3.

    Args:
        video_path (str): Path to the input video file (.mp4, .avi, .mov, etc.)
        output_path (str, optional): Path for the output MP3 file.
                                   If None, uses the same name as video with .mp3 extension
        overwrite (bool): Whether to overwrite existing output file. Default is False.

    Returns:
        str: Path to the generated MP3 file

    Raises:
        FileNotFoundError: If the input video file doesn't exist
        FileExistsError: If output file exists and overwrite is False
        Exception: If ffmpeg conversion fails
    """
    # Validate input file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Generate output path if not provided
    if output_path is None:
        video_stem = Path(video_path).stem
        video_dir = Path(video_path).parent
        output_path = video_dir / f"{video_stem}.mp3"
    else:
        output_path = Path(output_path)

    # Check if output file already exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Set overwrite=True to replace it."
        )

    try:
        # Extract audio using ffmpeg
        stream = ffmpeg.input(video_path)
        audio = stream.audio

        # Configure output with MP3 codec and quality settings
        out = ffmpeg.output(
            audio,
            str(output_path),
            acodec="mp3",
            audio_bitrate="192k",
            ar=44100,  # Sample rate
        )

        # Run the conversion
        if overwrite:
            out = ffmpeg.overwrite_output(out)

        ffmpeg.run(out, quiet=True)

        print(f"Successfully extracted audio to: {output_path}")
        return str(output_path)

    except ffmpeg.Error as e:
        raise Exception(f"FFmpeg error during conversion: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during audio extraction: {e}")


def extract_audio_to_mp3_simple(video_path):
    """
    Simplified version that extracts audio to MP3 with default settings.

    Args:
        video_path (str): Path to the input video file

    Returns:
        str: Path to the generated MP3 file
    """
    return extract_audio_to_mp3(video_path, overwrite=True)


# Example usage
if __name__ == "__main__":
    # Example usage - uncomment and modify paths as needed
    video_file = "annoyed.mp4"
    mp3_file = extract_audio_to_mp3(video_file)
    print(f"Audio extracted to: {mp3_file}")
