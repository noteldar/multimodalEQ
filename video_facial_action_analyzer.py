import cv2
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import logging
from datetime import datetime
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import facetorch
    from facetorch import FaceAnalyzer
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
except ImportError:
    logger.error("facetorch not installed. Install with: pip install facetorch")
    raise


class VideoFacialActionAnalyzer:
    """
    A class to analyze MP4 videos for facial action detection using facetorch.

    Based on the facétorch library which provides:
    - Face detection
    - Facial Action Units (AU) detection
    - Facial expression recognition
    - Face alignment and landmarks
    """

    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the analyzer with facetorch configuration.

        Args:
            config_path: Path to custom facetorch config file (uses default if None)
            device: Device to run analysis on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)

        # Initialize facetorch analyzer
        try:
            # Create a simple configuration that should work
            from omegaconf import OmegaConf

            # Try to use a basic configuration that includes the essential components
            cfg = OmegaConf.create(
                {
                    "logger": {"_target_": "facetorch.logger.Logger", "level": "INFO"},
                    "reader": {"_target_": "facetorch.reader.ImageReader"},
                    "detector": {
                        "_target_": "facetorch.detector.RetinaFaceDetector",
                        "device": self.device,
                    },
                    "unifier": {"_target_": "facetorch.unifier.FaceUnifier"},
                    "predictor": {
                        "au": {
                            "_target_": "facetorch.predictor.AUPredictor",
                            "device": self.device,
                        },
                        "fer": {
                            "_target_": "facetorch.predictor.FERPredictor",
                            "device": self.device,
                        },
                    },
                }
            )

            self.analyzer = FaceAnalyzer(cfg)
            logger.info(
                f"FaceAnalyzer initialized with device preference: {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalyzer: {e}")
            # Try even simpler fallback
            try:
                logger.info("Attempting simplified initialization...")
                cfg = OmegaConf.create(
                    {"logger": {"_target_": "facetorch.logger.Logger", "level": "INFO"}}
                )
                self.analyzer = FaceAnalyzer(cfg)
                logger.info("FaceAnalyzer initialized with minimal configuration")
            except Exception as fallback_e:
                logger.error(f"Fallback initialization also failed: {fallback_e}")
                raise

    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for processing."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def extract_frames(self, video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
        """
        Extract frames from MP4 video.

        Args:
            video_path: Path to the MP4 video file
            sample_rate: Extract every nth frame (1 = all frames)

        Returns:
            List of extracted frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frames = []
        frame_count = 0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Convert BGR to RGB (facetorch expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(
                    f"Extracted {len(frames)} frames ({frame_count}/{total_frames})"
                )

        cap.release()
        logger.info(f"Total frames extracted: {len(frames)}")
        return frames

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for facial actions using facetorch.

        Args:
            frame: Input frame as RGB numpy array

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Run facetorch analysis on the frame
            # The analyzer will detect faces and run all configured predictors
            response = self.analyzer.run(
                img=frame,
                batch_size=8,  # Process multiple faces in batch
                fix_img_size=True,
                return_img_data=False,  # Don't return processed images to save memory
                include_tensors=True,  # Include raw prediction tensors
            )

            results = {
                "timestamp": datetime.now().isoformat(),
                "faces_detected": len(response.faces),
                "faces": [],
            }

            # Process each detected face
            for face_idx, face in enumerate(response.faces):
                face_result = {
                    "face_id": face_idx,
                    "bbox": {
                        "x": float(face.loc[0]) if hasattr(face, "loc") else None,
                        "y": float(face.loc[1]) if hasattr(face, "loc") else None,
                        "width": float(face.loc[2]) if hasattr(face, "loc") else None,
                        "height": float(face.loc[3]) if hasattr(face, "loc") else None,
                    },
                    "confidence": float(face.score) if hasattr(face, "score") else None,
                }

                # Extract Action Units (AU) predictions
                if hasattr(face, "preds") and "au" in face.preds:
                    au_preds = face.preds["au"]
                    face_result["action_units"] = {
                        "predictions": (
                            au_preds.logits.tolist()
                            if hasattr(au_preds, "logits")
                            else []
                        ),
                        "labels": (
                            au_preds.labels if hasattr(au_preds, "labels") else []
                        ),
                        "scores": (
                            au_preds.scores if hasattr(au_preds, "scores") else []
                        ),
                    }

                # Extract other facial analysis results
                if hasattr(face, "preds"):
                    # Facial Expression Recognition (FER)
                    if "fer" in face.preds:
                        fer_preds = face.preds["fer"]
                        face_result["expression"] = {
                            "prediction": (
                                fer_preds.label if hasattr(fer_preds, "label") else None
                            ),
                            "confidence": (
                                float(fer_preds.score)
                                if hasattr(fer_preds, "score")
                                else None
                            ),
                            "all_scores": (
                                fer_preds.scores if hasattr(fer_preds, "scores") else []
                            ),
                        }

                    # Valence-Arousal
                    if "va" in face.preds:
                        va_preds = face.preds["va"]
                        face_result["valence_arousal"] = {
                            "valence": (
                                float(va_preds.valence)
                                if hasattr(va_preds, "valence")
                                else None
                            ),
                            "arousal": (
                                float(va_preds.arousal)
                                if hasattr(va_preds, "arousal")
                                else None
                            ),
                        }

                # Extract landmarks if available
                if hasattr(face, "landmarks"):
                    face_result["landmarks"] = (
                        face.landmarks.tolist() if face.landmarks is not None else []
                    )

                results["faces"].append(face_result)

            return results

        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "faces_detected": 0,
                "faces": [],
            }

    def analyze_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 1,
        save_format: str = "json",
    ) -> Dict:
        """
        Analyze complete MP4 video for facial actions.

        Args:
            video_path: Path to MP4 video file
            output_path: Path to save results (optional)
            sample_rate: Analyze every nth frame
            save_format: Output format ('json', 'csv', 'both')

        Returns:
            Dictionary containing complete analysis results
        """
        logger.info(f"Starting video analysis: {video_path}")

        # Extract frames
        frames = self.extract_frames(video_path, sample_rate)

        if not frames:
            raise ValueError("No frames extracted from video")

        # Analyze each frame
        all_results = {
            "video_info": {
                "path": video_path,
                "total_frames_analyzed": len(frames),
                "sample_rate": sample_rate,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "frame_results": [],
        }

        for frame_idx, frame in enumerate(frames):
            logger.info(f"Analyzing frame {frame_idx + 1}/{len(frames)}")

            frame_result = self.analyze_frame(frame)
            frame_result["frame_index"] = frame_idx
            frame_result["frame_timestamp"] = (
                frame_idx * sample_rate / 30.0
            )  # Assume 30 FPS

            all_results["frame_results"].append(frame_result)

        # Save results if output path provided
        if output_path:
            self.save_results(all_results, output_path, save_format)

        logger.info("Video analysis completed")
        return all_results

    def save_results(self, results: Dict, output_path: str, format_type: str = "json"):
        """Save analysis results to file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if format_type in ["json", "both"]:
            json_path = str(Path(output_path).with_suffix(".json"))
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {json_path}")

        if format_type in ["csv", "both"]:
            csv_path = str(Path(output_path).with_suffix(".csv"))
            self._save_to_csv(results, csv_path)
            logger.info(f"Results saved to: {csv_path}")

    def _save_to_csv(self, results: Dict, csv_path: str):
        """Convert results to CSV format."""
        rows = []

        for frame_result in results["frame_results"]:
            for face in frame_result.get("faces", []):
                row = {
                    "frame_index": frame_result.get("frame_index"),
                    "frame_timestamp": frame_result.get("frame_timestamp"),
                    "face_id": face.get("face_id"),
                    "bbox_x": face.get("bbox", {}).get("x"),
                    "bbox_y": face.get("bbox", {}).get("y"),
                    "bbox_width": face.get("bbox", {}).get("width"),
                    "bbox_height": face.get("bbox", {}).get("height"),
                    "face_confidence": face.get("confidence"),
                }

                # Add action units data
                if "action_units" in face:
                    au_data = face["action_units"]
                    if "labels" in au_data and "scores" in au_data:
                        for label, score in zip(au_data["labels"], au_data["scores"]):
                            row[f"AU_{label}"] = score

                # Add expression data
                if "expression" in face:
                    row["expression"] = face["expression"].get("prediction")
                    row["expression_confidence"] = face["expression"].get("confidence")

                # Add valence-arousal data
                if "valence_arousal" in face:
                    row["valence"] = face["valence_arousal"].get("valence")
                    row["arousal"] = face["valence_arousal"].get("arousal")

                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)


def analyze_video_facial_actions(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 1,
    config_path: Optional[str] = None,
    device: str = "auto",
) -> Dict:
    """
    Main function to analyze MP4 video for facial action detection using facetorch.

    Args:
        video_path: Path to the MP4 video file
        output_path: Path to save analysis results (optional)
        sample_rate: Analyze every nth frame (1 = all frames)
        config_path: Path to custom facetorch config (uses default if None)
        device: Device for processing ('auto', 'cpu', 'cuda')

    Returns:
        Dictionary containing complete facial action analysis results

    Example:
        >>> results = analyze_video_facial_actions(
        ...     video_path="sample_video.mp4",
        ...     output_path="results/facial_analysis",
        ...     sample_rate=5,  # Analyze every 5th frame
        ...     device="cuda"
        ... )
        >>> print(f"Analyzed {len(results['frame_results'])} frames")
    """
    analyzer = VideoFacialActionAnalyzer(config_path=config_path, device=device)
    return analyzer.analyze_video(
        video_path=video_path,
        output_path=output_path,
        sample_rate=sample_rate,
        save_format="both",  # Save as both JSON and CSV
    )


if __name__ == "__main__":
    # Example usage
    video_file = "henry.mp4"  # Replace with your video path
    output_dir = "facial_analysis_results"

    try:
        results = analyze_video_facial_actions(
            video_path=video_file,
            output_path=output_dir,
            sample_rate=10,  # Analyze every 10th frame for faster processing
            device="auto",
        )

        print(f"Analysis completed!")
        print(
            f"Total frames analyzed: {results['video_info']['total_frames_analyzed']}"
        )
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
