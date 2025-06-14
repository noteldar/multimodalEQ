# multimodalEQ

## Video Transcription

This project includes functionality to transcribe MP4 videos using OpenAI's Whisper model.

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: You may also need to install FFmpeg on your system:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt update && sudo apt install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

### Usage

#### From Python scripts:

```python
from get_transcript import get_video_transcript, get_video_transcript_with_timestamps

# Basic transcription (returns text only)
transcript = get_video_transcript("path/to/video.mp4")
print(transcript)

# Transcription with timestamps (returns full result dict)
result = get_video_transcript_with_timestamps("path/to/video.mp4")
print(result["text"])  # Full transcript
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

#### From command line:

```bash
python get_transcript.py path/to/video.mp4
```

### Model Options

You can specify different Whisper model sizes for different accuracy/speed tradeoffs:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: Higher accuracy
- `large`: Best accuracy, slowest

```python
transcript = get_video_transcript("video.mp4", model_size="large")
```
