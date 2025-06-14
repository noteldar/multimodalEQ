# Facial Action Analysis for MP4 Videos using FaceTorch

This repository contains comprehensive tools for analyzing facial actions in MP4 videos using the [facetorch library](https://github.com/tomas-gajarsky/facetorch). The analysis includes Facial Action Units (AU) detection, emotion recognition, and valence-arousal analysis.

## Features

### Core Analysis Capabilities
- **Facial Action Units (AU) Detection**: Based on FACS (Facial Action Coding System)
- **Facial Expression Recognition**: Emotion classification (happy, sad, angry, etc.)
- **Valence-Arousal Analysis**: Dimensional emotion analysis
- **Face Detection & Tracking**: Multi-face detection with confidence scores
- **Landmark Detection**: Key facial landmarks extraction
- **Face Embeddings**: Identity-related feature vectors

### Output Formats
- **JSON**: Complete detailed results with all metadata
- **CSV**: Summary tables for statistical analysis
- **Text Reports**: Human-readable analysis summaries
- **Action Units Timeline**: Detailed AU activation over time

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install facetorch:
```bash
pip install facetorch
```

## Usage

### Basic Usage

```python
from facetorch_video_analyzer import analyze_video_facial_actions

# Analyze a video with default settings
results = analyze_video_facial_actions(
    video_path="sample_video.mp4",
    output_path="results/",
    sample_rate=5  # Analyze every 5th frame
)

print(f"Analyzed {len(results['frame_results'])} frames")
```

### Advanced Usage with Enhanced Analyzer

```python
from enhanced_facetorch_analyzer import FacialActionVideoAnalyzer

# Initialize analyzer
analyzer = FacialActionVideoAnalyzer()

# Analyze video with comprehensive output
results = analyzer.analyze_video(
    video_path="interview.mp4",
    output_dir="results/",
    sample_rate=10  # Every 10th frame for faster processing
)

# Access summary statistics
stats = results['summary_statistics']
print(f"Most common emotion: {stats['most_common_emotion']}")
print(f"Average valence: {stats['average_valence']:.3f}")
print(f"Most active AUs: {stats['most_active_action_units'][:3]}")
```

### Command Line Usage

```bash
# Basic analysis
python enhanced_facetorch_analyzer.py video.mp4 --output results/

# Custom sample rate (faster processing)
python enhanced_facetorch_analyzer.py video.mp4 --sample-rate 10 --output results/

# Verbose logging
python enhanced_facetorch_analyzer.py video.mp4 --verbose --output results/
```

## Facial Action Units (FACS)

The system detects and analyzes the following Action Units based on the Facial Action Coding System:

| AU Code | Description | Muscle Movement |
|---------|-------------|-----------------|
| AU1 | Inner Brow Raiser | Frontalis medialis |
| AU2 | Outer Brow Raiser | Frontalis lateralis |
| AU4 | Brow Lowerer | Depressor glabellae |
| AU5 | Upper Lid Raiser | Levator palpebrae |
| AU6 | Cheek Raiser | Orbicularis oculi |
| AU7 | Lid Tightener | Orbicularis oculi |
| AU9 | Nose Wrinkler | Levator labii superioris |
| AU10 | Upper Lip Raiser | Levator labii superioris |
| AU12 | Lip Corner Puller | Zygomaticus major |
| AU14 | Dimpler | Buccinator |
| AU15 | Lip Corner Depressor | Depressor anguli oris |
| AU17 | Chin Raiser | Mentalis |
| AU20 | Lip Stretcher | Risorius |
| AU23 | Lip Tightener | Orbicularis oris |
| AU25 | Lips Part | Depressor labii |
| AU26 | Jaw Drop | Masseter |
| AU45 | Blink | Orbicularis oculi |

## Output Structure

### JSON Output Format

```json
{
  "analysis_metadata": {
    "video_path": "sample_video.mp4",
    "analysis_start_time": "2024-01-15T10:30:00",
    "processing_duration_seconds": 45.2,
    "frames_processed": 150
  },
  "video_metadata": {
    "total_frames": 750,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_seconds": 25.0
  },
  "frame_analyses": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "faces_detected": 1,
      "faces": [
        {
          "face_id": 0,
          "detection_confidence": 0.95,
          "action_units": {
            "active_action_units": ["AU6", "AU12"],
            "interpretations": {
              "AU12": {
                "score": 0.73,
                "active": true,
                "description": "Lip Corner Puller (Smile)",
                "intensity": "Moderate"
              }
            }
          },
          "expression": {
            "predicted_emotion": "happy",
            "confidence": 0.87
          },
          "valence_arousal": {
            "valence": 0.45,
            "arousal": 0.32,
            "emotional_quadrant": "Happy/Excited"
          }
        }
      ]
    }
  ],
  "summary_statistics": {
    "total_frames": 150,
    "frames_with_faces": 148,
    "most_common_emotion": "happy",
    "average_valence": 0.23,
    "most_active_action_units": [["AU12", 89], ["AU6", 67]]
  }
}
```

### CSV Output Files

1. **Summary CSV** (`facial_analysis_summary_TIMESTAMP.csv`):
   - Frame-by-frame summary with key metrics
   - Emotion predictions and confidence scores
   - Valence-arousal values
   - Active Action Units count

2. **Action Units Timeline** (`action_units_timeline_TIMESTAMP.csv`):
   - Detailed AU activation timeline
   - Individual AU scores and intensities
   - Frame-by-frame AU descriptions

## Performance Considerations

### Processing Speed
- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Processes multiple faces simultaneously
- **Memory Optimization**: Efficient frame processing to handle long videos
- **Sampling Rate**: Adjustable frame sampling for speed vs. accuracy trade-off

### Sample Processing Times (Tesla T4 GPU)
- **1080p 30fps video**: ~486ms per frame (4 faces)
- **Batch processing**: ~1845ms for 25 faces
- **Memory usage**: ~2-4GB GPU memory depending on video resolution

### Recommended Settings
- **High accuracy**: `sample_rate=1` (every frame)
- **Balanced**: `sample_rate=5` (every 5th frame)
- **Fast processing**: `sample_rate=10` (every 10th frame)

## Example Applications

### 1. Interview Analysis
```python
# Analyze job interview for emotional states
results = analyzer.analyze_video(
    video_path="job_interview.mp4",
    output_dir="interview_analysis/",
    sample_rate=5
)

# Extract key insights
emotions = [face['expression']['predicted_emotion'] 
           for frame in results['frame_analyses'] 
           for face in frame['faces'] 
           if 'expression' in face]

print(f"Emotion distribution: {Counter(emotions)}")
```

### 2. Therapy Session Analysis
```python
# Analyze therapy session for affect patterns
results = analyzer.analyze_video(
    video_path="therapy_session.mp4",
    output_dir="therapy_analysis/",
    sample_rate=3
)

# Track valence over time
valence_timeline = []
for frame in results['frame_analyses']:
    for face in frame['faces']:
        if 'valence_arousal' in face:
            valence_timeline.append({
                'timestamp': frame['timestamp_seconds'],
                'valence': face['valence_arousal']['valence']
            })
```

### 3. Educational Content Analysis
```python
# Analyze student engagement in educational videos
results = analyzer.analyze_video(
    video_path="lecture.mp4",
    output_dir="engagement_analysis/",
    sample_rate=15  # Lower frequency for long lectures
)

# Identify attention patterns
attention_markers = []
for frame in results['frame_analyses']:
    for face in frame['faces']:
        if 'action_units' in face:
            # Look for attention-related AUs
            if 'AU5' in face['action_units']['active_action_units']:  # Upper lid raiser
                attention_markers.append(frame['timestamp_seconds'])
```

## Troubleshooting

### Common Issues

1. **Installation Problems**:
   ```bash
   # If facetorch installation fails
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install facetorch
   ```

2. **Memory Issues**:
   - Reduce batch size in analyzer configuration
   - Increase sample rate (process fewer frames)
   - Use CPU instead of GPU for very long videos

3. **Video Format Issues**:
   ```bash
   # Convert video to compatible format
   ffmpeg -i input_video.mov -c:v libx264 -c:a aac output_video.mp4
   ```

### Error Handling

The scripts include comprehensive error handling:
- Invalid video file formats
- Missing faces in frames
- GPU memory overflow
- Corrupted video files

## Configuration

### Custom FaceTorch Configuration

Create a custom YAML configuration file for specific models:

```yaml
# custom_config.yaml
analyzer:
  detector:
    name: "RetinaFace"
    model: "resnet50"
  predictor:
    au:
      name: "EfficientNet-B2"
      model: "efficientnet_b2_8"
    fer:
      name: "EfficientNet-B2" 
      model: "efficientnet_b2_8"
```

Use with:
```python
analyzer = FacialActionVideoAnalyzer(config_path="custom_config.yaml")
```

## Contributing

Based on the excellent [facetorch library](https://github.com/tomas-gajarsky/facetorch) by Tomas Gajarsky. 

## Citation

If you use this code in your research, please cite both this work and the underlying facetorch library:

```bibtex
@misc{facetorch,
    author = {Gajarsky, Tomas},
    title = {Facetorch: A Python Library for Analyzing Faces Using PyTorch},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/tomas-gajarsky/facetorch}}
}
```

## License

This project uses the same license as the underlying facetorch library (Apache-2.0). 