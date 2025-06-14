#!/usr/bin/env python3
"""
Video Facial Action Analyzer using facetorch

This script analyzes MP4 videos for facial action detection using the facetorch library.
Based on the facetorch repository: https://github.com/tomas-gajarsky/facetorch

Features:
- Face detection and tracking
- Facial Action Units (AU) detection
- Facial expression recognition
- Valence-Arousal analysis
- Frame-by-frame analysis with configurable sampling
- Output in JSON and CSV formats
"""

import cv2
import os
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import facetorch
    from facetorch import FaceAnalyzer
except ImportError:
    logger.error("facetorch not installed. Install with: pip install facetorch")
    raise ImportError("Please install facetorch: pip install facetorch")


class VideoFacialActionAnalyzer:
    """
    Analyzes MP4 videos for facial actions using facetorch library.

    The facetorch library provides comprehensive facial analysis including:
    - Face detection using various models (RetinaFace, MTCNN, etc.)
    - Facial Action Units (AU) detection
    - Facial Expression Recognition (FER)
    - Face verification and embeddings
    - 3D face alignment
    - Deepfake detection
    """

    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the facial action analyzer.

        Args:
            config_path: Path to facetorch config YAML file (uses default if None)
            device: Processing device ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self._initialize_analyzer(config_path)

    def _setup_device(self, device: str) -> str:
        """Setup processing device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_analyzer(self, config_path: Optional[str] = None):
        """Initialize facetorch FaceAnalyzer."""
        try:
            if config_path and os.path.exists(config_path):
                self.analyzer = FaceAnalyzer(cfg_path=config_path)
            else:
                # Use default configuration which includes AU detection
                self.analyzer = FaceAnalyzer()

            logger.info(f"FaceAnalyzer initialized successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalyzer: {e}")
            raise RuntimeError(f"Cannot initialize facetorch analyzer: {e}")

    def extract_video_frames(
        self, video_path: str, sample_rate: int = 1
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from MP4 video.

        Args:
            video_path: Path to MP4 video file
            sample_rate: Extract every nth frame (1 = all frames)

        Returns:
            Tuple of (frames_list, video_metadata)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video metadata
        metadata = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": None,
        }

        if metadata["fps"] > 0:
            metadata["duration_seconds"] = metadata["total_frames"] / metadata["fps"]

        logger.info(
            f"Video info: {metadata['total_frames']} frames, {metadata['fps']:.2f} FPS, "
            f"{metadata['width']}x{metadata['height']}"
        )

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Convert BGR to RGB (facetorch expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1

            if frame_count % 500 == 0:
                logger.info(
                    f"Processed {frame_count}/{metadata['total_frames']} frames"
                )

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")

        return frames, metadata

    def analyze_single_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Analyze a single frame for facial actions.

        Args:
            frame: RGB frame as numpy array
            frame_idx: Frame index number

        Returns:
            Dictionary with analysis results
        """
        try:
            # Run facetorch analysis
            response = self.analyzer.run(
                img=frame,
                batch_size=8,
                fix_img_size=True,
                return_img_data=False,
                include_tensors=True,
            )

            frame_results = {
                "frame_index": frame_idx,
                "analysis_timestamp": datetime.now().isoformat(),
                "faces_detected": len(response.faces),
                "faces": [],
            }

            # Process each detected face
            for face_idx, face in enumerate(response.faces):
                face_data = self._extract_face_data(face, face_idx)
                frame_results["faces"].append(face_data)

            return frame_results

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_idx}: {e}")
            return {
                "frame_index": frame_idx,
                "error": str(e),
                "faces_detected": 0,
                "faces": [],
            }

    def _extract_face_data(self, face, face_idx: int) -> Dict:
        """Extract comprehensive data from a detected face."""
        face_data = {
            "face_id": face_idx,
            "detection_confidence": (
                float(face.score) if hasattr(face, "score") else None
            ),
        }

        # Bounding box coordinates
        if hasattr(face, "loc") and face.loc is not None:
            face_data["bounding_box"] = {
                "x": float(face.loc[0]),
                "y": float(face.loc[1]),
                "width": float(face.loc[2] - face.loc[0]),
                "height": float(face.loc[3] - face.loc[1]),
            }

        # Extract predictions if available
        if hasattr(face, "preds") and face.preds:

            # Action Units (AU) - Primary focus
            if "au" in face.preds:
                au_pred = face.preds["au"]
                face_data["action_units"] = self._extract_action_units(au_pred)

            # Facial Expression Recognition
            if "fer" in face.preds:
                fer_pred = face.preds["fer"]
                face_data["expression"] = {
                    "predicted_emotion": (
                        fer_pred.label if hasattr(fer_pred, "label") else None
                    ),
                    "confidence": (
                        float(fer_pred.score) if hasattr(fer_pred, "score") else None
                    ),
                    "all_emotions": (
                        fer_pred.scores if hasattr(fer_pred, "scores") else {}
                    ),
                }

            # Valence-Arousal
            if "va" in face.preds:
                va_pred = face.preds["va"]
                face_data["valence_arousal"] = {
                    "valence": (
                        float(va_pred.valence) if hasattr(va_pred, "valence") else None
                    ),
                    "arousal": (
                        float(va_pred.arousal) if hasattr(va_pred, "arousal") else None
                    ),
                }

            # Face embeddings for identity
            if "embed" in face.preds:
                embed_pred = face.preds["embed"]
                if hasattr(embed_pred, "logits"):
                    face_data["face_embedding"] = (
                        embed_pred.logits.cpu().numpy().tolist()
                    )

        # Facial landmarks
        if hasattr(face, "landmarks") and face.landmarks is not None:
            face_data["landmarks"] = face.landmarks.tolist()

        return face_data

    def _extract_action_units(self, au_pred) -> Dict:
        """Extract Action Units data from prediction."""
        au_data = {}

        if hasattr(au_pred, "logits") and au_pred.logits is not None:
            au_data["raw_predictions"] = au_pred.logits.cpu().numpy().tolist()

        if hasattr(au_pred, "labels") and au_pred.labels:
            au_data["labels"] = au_pred.labels

        if hasattr(au_pred, "scores") and au_pred.scores:
            au_data["scores"] = {
                label: float(score) for label, score in au_pred.scores.items()
            }

        # Common Action Units mapping (based on FACS - Facial Action Coding System)
        au_meanings = {
            "AU1": "Inner Brow Raiser",
            "AU2": "Outer Brow Raiser",
            "AU4": "Brow Lowerer",
            "AU5": "Upper Lid Raiser",
            "AU6": "Cheek Raiser",
            "AU7": "Lid Tightener",
            "AU9": "Nose Wrinkler",
            "AU10": "Upper Lip Raiser",
            "AU12": "Lip Corner Puller",
            "AU14": "Dimpler",
            "AU15": "Lip Corner Depressor",
            "AU17": "Chin Raiser",
            "AU20": "Lip Stretcher",
            "AU23": "Lip Tightener",
            "AU25": "Lips Part",
            "AU26": "Jaw Drop",
            "AU45": "Blink",
        }

        if "scores" in au_data:
            au_data["interpreted_actions"] = {
                au: {
                    "score": au_data["scores"].get(au, 0.0),
                    "description": au_meanings.get(au, "Unknown Action Unit"),
                    "active": au_data["scores"].get(au, 0.0) > 0.5,
                }
                for au in au_meanings.keys()
                if au in au_data["scores"]
            }

        return au_data

    def analyze_video(
        self, video_path: str, output_path: Optional[str] = None, sample_rate: int = 1
    ) -> Dict:
        """
        Analyze complete video for facial actions.

        Args:
            video_path: Path to MP4 video
            output_path: Output directory for results
            sample_rate: Analyze every nth frame

        Returns:
            Complete analysis results
        """
        logger.info(f"Starting facial action analysis: {video_path}")

        # Extract frames
        frames, video_metadata = self.extract_video_frames(video_path, sample_rate)

        if not frames:
            raise ValueError("No frames extracted from video")

        # Complete analysis results
        analysis_results = {
            "video_metadata": {
                **video_metadata,
                "source_path": video_path,
                "sample_rate": sample_rate,
                "frames_analyzed": len(frames),
                "analysis_start_time": datetime.now().isoformat(),
            },
            "frame_analyses": [],
        }

        # Analyze each frame
        for idx, frame in enumerate(frames):
            if idx % 50 == 0:
                logger.info(f"Analyzing frame {idx + 1}/{len(frames)}")

            frame_result = self.analyze_single_frame(frame, idx)

            # Add timing information
            if video_metadata["fps"] > 0:
                frame_result["timestamp_seconds"] = (
                    idx * sample_rate
                ) / video_metadata["fps"]

            analysis_results["frame_analyses"].append(frame_result)

        analysis_results["video_metadata"][
            "analysis_end_time"
        ] = datetime.now().isoformat()

        # Save results if output path provided
        if output_path:
            self._save_analysis_results(analysis_results, output_path)

        logger.info(f"Video analysis completed. Analyzed {len(frames)} frames.")
        return analysis_results

    def _save_analysis_results(self, results: Dict, output_path: str):
        """Save analysis results in multiple formats."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save complete results as JSON
        json_file = output_dir / "facial_analysis_complete.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Complete results saved: {json_file}")

        # Save summary as CSV
        csv_file = output_dir / "facial_analysis_summary.csv"
        self._create_summary_csv(results, csv_file)
        logger.info(f"Summary CSV saved: {csv_file}")

        # Save action units timeline as CSV
        au_csv_file = output_dir / "action_units_timeline.csv"
        self._create_action_units_csv(results, au_csv_file)
        logger.info(f"Action units timeline saved: {au_csv_file}")

    def _create_summary_csv(self, results: Dict, csv_path: Path):
        """Create summary CSV with key metrics per frame."""
        rows = []

        for frame_data in results["frame_analyses"]:
            base_row = {
                "frame_index": frame_data["frame_index"],
                "timestamp_seconds": frame_data.get("timestamp_seconds", 0),
                "faces_detected": frame_data["faces_detected"],
            }

            if frame_data["faces"]:
                for face in frame_data["faces"]:
                    row = base_row.copy()
                    row["face_id"] = face["face_id"]
                    row["detection_confidence"] = face.get("detection_confidence")

                    # Expression data
                    if "expression" in face:
                        row["predicted_emotion"] = face["expression"].get(
                            "predicted_emotion"
                        )
                        row["emotion_confidence"] = face["expression"].get("confidence")

                    # Valence-Arousal
                    if "valence_arousal" in face:
                        row["valence"] = face["valence_arousal"].get("valence")
                        row["arousal"] = face["valence_arousal"].get("arousal")

                    rows.append(row)
            else:
                rows.append(base_row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

    def _create_action_units_csv(self, results: Dict, csv_path: Path):
        """Create detailed Action Units timeline CSV."""
        rows = []

        for frame_data in results["frame_analyses"]:
            for face in frame_data.get("faces", []):
                if (
                    "action_units" in face
                    and "interpreted_actions" in face["action_units"]
                ):
                    base_row = {
                        "frame_index": frame_data["frame_index"],
                        "timestamp_seconds": frame_data.get("timestamp_seconds", 0),
                        "face_id": face["face_id"],
                    }

                    # Add each action unit as a column
                    for au_name, au_info in face["action_units"][
                        "interpreted_actions"
                    ].items():
                        row = base_row.copy()
                        row["action_unit"] = au_name
                        row["description"] = au_info["description"]
                        row["score"] = au_info["score"]
                        row["active"] = au_info["active"]
                        rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)


def analyze_video_facial_actions(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 1,
    device: str = "auto",
) -> Dict:
    """
    Main function to analyze MP4 video for facial action detection.

    Args:
        video_path: Path to MP4 video file
        output_path: Directory to save results (optional)
        sample_rate: Analyze every nth frame (1 = all frames)
        device: Processing device ('auto', 'cpu', 'cuda')

    Returns:
        Dictionary containing complete facial action analysis

    Example:
        >>> results = analyze_video_facial_actions(
        ...     video_path="interview.mp4",
        ...     output_path="results/",
        ...     sample_rate=5,  # Every 5th frame
        ...     device="cuda"
        ... )
    """
    analyzer = VideoFacialActionAnalyzer(device=device)
    return analyzer.analyze_video(video_path, output_path, sample_rate)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MP4 video for facial actions using facetorch"
    )
    parser.add_argument("video_path", help="Path to MP4 video file")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=int,
        default=5,
        help="Analyze every nth frame (default: 5)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Processing device (default: auto)",
    )

    args = parser.parse_args()

    try:
        results = analyze_video_facial_actions(
            video_path=args.video_path,
            output_path=args.output,
            sample_rate=args.sample_rate,
            device=args.device,
        )

        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Frames analyzed: {results['video_metadata']['frames_analyzed']}")
        print(
            f"🎬 Video duration: {results['video_metadata'].get('duration_seconds', 'Unknown'):.2f}s"
        )

        if args.output:
            print(f"💾 Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise
