from faster_whisper import WhisperModel
import os
from pathlib import Path


def get_video_transcript(video_path: str, model_size: str = "base") -> str:
    """
    Transcribe an MP4 video file using faster-whisper.

    Args:
        video_path (str): Path to the MP4 video file
        model_size (str): Whisper model size to use. Options: tiny, base, small, medium, large
                         Default is "large" for best accuracy

    Returns:
        str: The transcript text of the video

    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the file is not a valid video format
        Exception: For other transcription errors
    """
    # Validate input file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check file extension
    video_path_obj = Path(video_path)
    if video_path_obj.suffix.lower() not in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    ]:
        raise ValueError(f"Unsupported video format: {video_path_obj.suffix}")

    try:
        # Load the faster-whisper model
        print(f"Loading Whisper model: {model_size}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Transcribe the video
        print(f"Transcribing video: {video_path}")
        segments, info = model.transcribe(video_path, beam_size=5)

        # Collect all transcript segments
        transcript_text = ""
        for segment in segments:
            transcript_text += segment.text + " "

        # Return the transcript text
        return transcript_text.strip()

    except Exception as e:
        raise Exception(f"Error transcribing video: {str(e)}")


def get_video_transcript_with_timestamps(
    video_path: str, model_size: str = "large"
) -> dict:
    """
    Transcribe an MP4 video file using faster-whisper with timestamps.

    Args:
        video_path (str): Path to the MP4 video file
        model_size (str): Whisper model size to use

    Returns:
        dict: Full transcription result including segments with timestamps
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_path_obj = Path(video_path)
    if video_path_obj.suffix.lower() not in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    ]:
        raise ValueError(f"Unsupported video format: {video_path_obj.suffix}")

    try:
        print(f"Loading Whisper model: {model_size}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        print(f"Transcribing video with timestamps: {video_path}")
        segments, info = model.transcribe(video_path, beam_size=5)
        
        # Format results similar to openai-whisper
        result = {
            "text": "",
            "segments": [],
            "language": info.language
        }
        
        for segment in segments:
            result["text"] += segment.text + " "
            result["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        result["text"] = result["text"].strip()
        return result

    except Exception as e:
        raise Exception(f"Error transcribing video: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        try:
            transcript = get_video_transcript(video_file)
            print("\n--- TRANSCRIPT ---")
            print(transcript)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python transcript.py <video_file_path>")
