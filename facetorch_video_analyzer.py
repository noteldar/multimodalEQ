import cv2
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from facetorch import FaceAnalyzer
except ImportError:
    raise ImportError("Please install facetorch: pip install facetorch")


def analyze_video_facial_actions(
    video_path: str, output_path: Optional[str] = None, sample_rate: int = 5
) -> Dict:
    """
    Analyze MP4 video for facial action detection using facetorch.

    Based on facetorch library: https://github.com/tomas-gajarsky/facetorch

    Args:
        video_path: Path to MP4 video file
        output_path: Directory to save results (optional)
        sample_rate: Analyze every nth frame (default: 5)

    Returns:
        Dictionary containing facial action analysis results
    """

    # Initialize facetorch analyzer
    logger.info("Initializing FaceAnalyzer...")
    analyzer = FaceAnalyzer()

    # Extract frames from video
    logger.info(f"Processing video: {video_path}")
    frames = extract_frames_from_video(video_path, sample_rate)

    results = {
        "video_info": {
            "path": video_path,
            "frames_analyzed": len(frames),
            "sample_rate": sample_rate,
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "frame_results": [],
    }

    # Analyze each frame
    for idx, frame in enumerate(frames):
        logger.info(f"Analyzing frame {idx + 1}/{len(frames)}")

        try:
            # Run facetorch analysis
            response = analyzer.run(
                img=frame,
                batch_size=8,
                fix_img_size=True,
                return_img_data=False,
                include_tensors=True,
            )

            frame_result = {
                "frame_index": idx,
                "timestamp": idx * sample_rate / 30.0,  # Assume 30 FPS
                "faces_detected": len(response.faces),
                "faces": [],
            }

            # Process each detected face
            for face_idx, face in enumerate(response.faces):
                face_data = {
                    "face_id": face_idx,
                    "confidence": float(face.score) if hasattr(face, "score") else None,
                }

                # Extract Action Units (AU) predictions
                if hasattr(face, "preds") and "au" in face.preds:
                    au_preds = face.preds["au"]
                    face_data["action_units"] = {
                        "predictions": (
                            au_preds.logits.tolist()
                            if hasattr(au_preds, "logits")
                            else []
                        ),
                        "labels": (
                            au_preds.labels if hasattr(au_preds, "labels") else []
                        ),
                        "scores": (
                            au_preds.scores if hasattr(au_preds, "scores") else {}
                        ),
                    }

                # Extract facial expression
                if hasattr(face, "preds") and "fer" in face.preds:
                    fer_preds = face.preds["fer"]
                    face_data["expression"] = {
                        "emotion": (
                            fer_preds.label if hasattr(fer_preds, "label") else None
                        ),
                        "confidence": (
                            float(fer_preds.score)
                            if hasattr(fer_preds, "score")
                            else None
                        ),
                    }

                frame_result["faces"].append(face_data)

            results["frame_results"].append(frame_result)

        except Exception as e:
            logger.error(f"Error processing frame {idx}: {e}")
            results["frame_results"].append(
                {"frame_index": idx, "error": str(e), "faces_detected": 0}
            )

    # Save results if output path provided
    if output_path:
        save_results(results, output_path)

    logger.info("Video analysis completed")
    return results


def extract_frames_from_video(
    video_path: str, sample_rate: int = 5
) -> List[np.ndarray]:
    """Extract frames from MP4 video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            # Convert BGR to RGB for facetorch
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames")
    return frames


def save_results(results: Dict, output_path: str):
    """Save analysis results to JSON file."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / "facial_analysis.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {json_file}")


if __name__ == "__main__":
    # Example usage
    video_file = "sample_video.mp4"  # Replace with your video path
    output_directory = "results/"

    try:
        results = analyze_video_facial_actions(
            video_path=video_file,
            output_path=output_directory,
            sample_rate=10,  # Analyze every 10th frame
        )

        print(f"Analysis completed! Processed {len(results['frame_results'])} frames")

    except Exception as e:
        print(f"Error: {e}")
