# Facial Action Analysis Usage Examples

This document provides comprehensive examples for using the facial action analysis tools with the facetorch library.

## Quick Start Example

```python
from facetorch_video_analyzer import analyze_video_facial_actions

# Basic video analysis
results = analyze_video_facial_actions(
    video_path="sample_video.mp4",
    output_path="results/",
    sample_rate=5
)

print(f"Analysis complete! Processed {len(results['frame_results'])} frames")
```

## Command Line Usage

```bash
# Basic analysis
python enhanced_facetorch_analyzer.py video.mp4 --output results/

# Analyze every 10th frame (faster)
python enhanced_facetorch_analyzer.py video.mp4 --sample-rate 10 --output results/

# Enable verbose logging
python enhanced_facetorch_analyzer.py video.mp4 --verbose --output results/
```

## Advanced Examples

### 1. Detailed Emotion Analysis

```python
from enhanced_facetorch_analyzer import FacialActionVideoAnalyzer

analyzer = FacialActionVideoAnalyzer()
results = analyzer.analyze_video("interview.mp4", "results/", sample_rate=5)

# Extract emotion timeline
emotions_timeline = []
for frame in results['frame_analyses']:
    for face in frame['faces']:
        if 'expression' in face:
            emotions_timeline.append({
                'timestamp': frame['timestamp_seconds'],
                'emotion': face['expression']['predicted_emotion'],
                'confidence': face['expression']['confidence']
            })

# Most common emotions
from collections import Counter
emotion_counts = Counter([e['emotion'] for e in emotions_timeline])
print(f"Most common emotion: {emotion_counts.most_common(1)[0]}")
```

### 2. Action Units Analysis

```python
# Extract Action Units data
au_data = []
for frame in results['frame_analyses']:
    for face in frame['faces']:
        if 'action_units' in face:
            active_aus = face['action_units'].get('active_action_units', [])
            au_data.append({
                'timestamp': frame['timestamp_seconds'],
                'active_aus': active_aus,
                'total_active': len(active_aus)
            })

# Find most active periods
high_activity_periods = [
    data for data in au_data 
    if data['total_active'] > 5
]

print(f"High activity periods: {len(high_activity_periods)}")
```

### 3. Valence-Arousal Tracking

```python
# Track emotional valence and arousal over time
va_timeline = []
for frame in results['frame_analyses']:
    for face in frame['faces']:
        if 'valence_arousal' in face:
            va_timeline.append({
                'timestamp': frame['timestamp_seconds'],
                'valence': face['valence_arousal']['valence'],
                'arousal': face['valence_arousal']['arousal'],
                'quadrant': face['valence_arousal']['emotional_quadrant']
            })

# Calculate average emotional state
avg_valence = sum(va['valence'] for va in va_timeline) / len(va_timeline)
avg_arousal = sum(va['arousal'] for va in va_timeline) / len(va_timeline)

print(f"Average emotional state: Valence={avg_valence:.3f}, Arousal={avg_arousal:.3f}")
```

## Facial Action Units (FACS) Interpretation

```python
# Interpret specific Action Units
def interpret_smile(au_data):
    """Detect genuine vs non-genuine smiles"""
    au_scores = au_data.get('scores', {})
    
    # Duchenne smile markers
    lip_corner_puller = au_scores.get('AU12', 0)  # Smile
    cheek_raiser = au_scores.get('AU6', 0)        # Eyes involved
    
    if lip_corner_puller > 0.5 and cheek_raiser > 0.5:
        return "Genuine smile (Duchenne)"
    elif lip_corner_puller > 0.5:
        return "Social smile"
    else:
        return "No smile"

# Apply to results
for frame in results['frame_analyses']:
    for face in frame['faces']:
        if 'action_units' in face:
            smile_type = interpret_smile(face['action_units'])
            print(f"Frame {frame['frame_index']}: {smile_type}")
```

## Performance Optimization

### GPU vs CPU Processing

```python
# Force CPU processing for compatibility
import torch
torch.cuda.set_device(-1)  # Disable CUDA

analyzer = FacialActionVideoAnalyzer()
results = analyzer.analyze_video("video.mp4", sample_rate=10)
```

### Memory Management for Long Videos

```python
# Process long videos in chunks
def process_long_video(video_path, chunk_duration=60):
    """Process video in 60-second chunks"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    chunk_frames = int(chunk_duration * fps)
    
    all_results = []
    
    for start_frame in range(0, total_frames, chunk_frames):
        end_frame = min(start_frame + chunk_frames, total_frames)
        
        # Extract chunk
        chunk_path = f"temp_chunk_{start_frame}.mp4"
        # ... extract chunk logic ...
        
        # Analyze chunk
        chunk_results = analyze_video_facial_actions(
            chunk_path, sample_rate=5
        )
        all_results.append(chunk_results)
    
    return all_results
```

## Data Export and Visualization

### Export to CSV for Analysis

```python
import pandas as pd

def export_summary_to_csv(results, output_path):
    """Export analysis summary to CSV"""
    rows = []
    
    for frame in results['frame_analyses']:
        for face in frame['faces']:
            row = {
                'frame_index': frame['frame_index'],
                'timestamp': frame['timestamp_seconds'],
                'face_id': face['face_id'],
                'emotion': face.get('expression', {}).get('predicted_emotion'),
                'emotion_confidence': face.get('expression', {}).get('confidence'),
                'valence': face.get('valence_arousal', {}).get('valence'),
                'arousal': face.get('valence_arousal', {}).get('arousal'),
                'active_aus_count': face.get('action_units', {}).get('total_active', 0)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

# Usage
df = export_summary_to_csv(results, "analysis_summary.csv")
print(df.head())
```

### Create Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_emotion_timeline(results):
    """Plot emotion changes over time"""
    emotions = []
    timestamps = []
    
    for frame in results['frame_analyses']:
        for face in frame['faces']:
            if 'expression' in face:
                emotions.append(face['expression']['predicted_emotion'])
                timestamps.append(frame['timestamp_seconds'])
    
    # Create timeline plot
    plt.figure(figsize=(12, 6))
    emotion_counts = {}
    for i, (time, emotion) in enumerate(zip(timestamps, emotions)):
        if emotion not in emotion_counts:
            emotion_counts[emotion] = []
        emotion_counts[emotion].append(time)
    
    colors = plt.cm.Set3(range(len(emotion_counts)))
    for i, (emotion, times) in enumerate(emotion_counts.items()):
        plt.scatter(times, [emotion] * len(times), 
                   color=colors[i], alpha=0.7, s=50)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Timeline')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('emotion_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create visualization
plot_emotion_timeline(results)
```

## Integration with Other Tools

### Save for Further Analysis

```python
# Save processed data for machine learning
def prepare_ml_features(results):
    """Prepare features for ML analysis"""
    features = []
    
    for frame in results['frame_analyses']:
        for face in frame['faces']:
            feature_vector = {
                'timestamp': frame['timestamp_seconds'],
                'face_id': face['face_id']
            }
            
            # Add AU features
            if 'action_units' in face and 'scores' in face['action_units']:
                for au_code, score in face['action_units']['scores'].items():
                    feature_vector[f'au_{au_code}'] = score
            
            # Add emotion features
            if 'expression' in face and 'all_emotions' in face['expression']:
                for emotion, score in face['expression']['all_emotions'].items():
                    feature_vector[f'emotion_{emotion}'] = score
            
            # Add VA features
            if 'valence_arousal' in face:
                feature_vector['valence'] = face['valence_arousal'].get('valence', 0)
                feature_vector['arousal'] = face['valence_arousal'].get('arousal', 0)
            
            features.append(feature_vector)
    
    return pd.DataFrame(features)

# Prepare and save features
features_df = prepare_ml_features(results)
features_df.to_csv('ml_features.csv', index=False)
```

## Error Handling and Debugging

### Common Issues and Solutions

```python
def robust_video_analysis(video_path, max_retries=3):
    """Robust video analysis with error handling"""
    
    for attempt in range(max_retries):
        try:
            results = analyze_video_facial_actions(
                video_path=video_path,
                sample_rate=5
            )
            return results
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if "CUDA out of memory" in str(e):
                print("Trying with CPU...")
                torch.cuda.empty_cache()
                # Force CPU processing
                
            elif "Video not found" in str(e):
                print("Check video path and format")
                break
                
            elif attempt == max_retries - 1:
                print("All attempts failed")
                raise
    
    return None

# Usage with error handling
results = robust_video_analysis("problematic_video.mp4")
```

## Configuration Examples

### Custom Model Configuration

```python
# Create custom analyzer with specific settings
class CustomFacialAnalyzer(FacialActionVideoAnalyzer):
    def __init__(self):
        super().__init__()
        # Custom initialization
        self.confidence_threshold = 0.7
        self.au_threshold = 0.5
    
    def filter_high_confidence_faces(self, faces):
        """Filter faces by confidence threshold"""
        return [
            face for face in faces 
            if face.get('detection_confidence', 0) > self.confidence_threshold
        ]

# Use custom analyzer
custom_analyzer = CustomFacialAnalyzer()
results = custom_analyzer.analyze_video("video.mp4", "results/")
```

This comprehensive set of examples should help you get started with facial action analysis using the facetorch library! 