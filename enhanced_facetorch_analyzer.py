#!/usr/bin/env python3
"""
Enhanced Facial Action Video Analyzer using facetorch

This script provides comprehensive facial action analysis for MP4 videos using the facetorch library.
Features include:
- Facial Action Units (AU) detection with FACS interpretation
- Facial expression recognition
- Valence-Arousal analysis
- Face detection and tracking
- Multiple output formats (JSON, CSV)
- Detailed logging and error handling

Based on: https://github.com/tomas-gajarsky/facetorch
"""

import cv2
import numpy as np
import pandas as pd
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("facial_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    from facetorch import FaceAnalyzer
except ImportError:
    logger.error("facetorch not installed. Install with: pip install facetorch")
    raise ImportError("Please install facetorch: pip install facetorch")


class FacialActionVideoAnalyzer:
    """
    Enhanced facial action analyzer for MP4 videos using facetorch.

    The facetorch library provides comprehensive facial analysis including:
    - Face detection using RetinaFace, MTCNN, and other SOTA models
    - Facial Action Units (AU) detection based on FACS (Facial Action Coding System)
    - Facial Expression Recognition (FER) for emotions
    - Valence-Arousal dimensional emotion analysis
    - 3D face alignment and landmark detection
    - Face verification and deepfake detection
    """

    # FACS Action Units mapping for interpretation
    ACTION_UNITS = {
        "AU1": "Inner Brow Raiser",
        "AU2": "Outer Brow Raiser",
        "AU4": "Brow Lowerer",
        "AU5": "Upper Lid Raiser",
        "AU6": "Cheek Raiser",
        "AU7": "Lid Tightener",
        "AU9": "Nose Wrinkler",
        "AU10": "Upper Lip Raiser",
        "AU12": "Lip Corner Puller (Smile)",
        "AU14": "Dimpler",
        "AU15": "Lip Corner Depressor",
        "AU17": "Chin Raiser",
        "AU20": "Lip Stretcher",
        "AU23": "Lip Tightener",
        "AU25": "Lips Part",
        "AU26": "Jaw Drop",
        "AU27": "Mouth Stretch",
        "AU45": "Blink",
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the analyzer with facetorch."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            if config_path and Path(config_path).exists():
                self.analyzer = FaceAnalyzer(cfg_path=config_path)
                logger.info(f"Loaded custom config: {config_path}")
            else:
                self.analyzer = FaceAnalyzer()
                logger.info("Using default facetorch configuration")
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalyzer: {e}")
            raise

    def extract_video_frames(
        self, video_path: str, sample_rate: int = 5
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from MP4 video with metadata.

        Args:
            video_path: Path to video file
            sample_rate: Extract every nth frame

        Returns:
            Tuple of (frames, metadata)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Extract video metadata
        metadata = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "sample_rate": sample_rate,
        }

        metadata["duration_seconds"] = (
            metadata["total_frames"] / metadata["fps"] if metadata["fps"] > 0 else 0
        )

        logger.info(
            f"Video: {metadata['total_frames']} frames, {metadata['fps']:.1f} FPS, "
            f"{metadata['width']}x{metadata['height']}, {metadata['duration_seconds']:.1f}s"
        )

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

            if frame_count % 1000 == 0:
                logger.info(
                    f"Extracted {len(frames)} frames ({frame_count}/{metadata['total_frames']})"
                )

        cap.release()
        metadata["frames_extracted"] = len(frames)
        logger.info(f"Frame extraction complete: {len(frames)} frames")

        return frames, metadata

    def analyze_frame(
        self, frame: np.ndarray, frame_idx: int, timestamp: float
    ) -> Dict:
        """
        Analyze single frame for facial actions.

        Args:
            frame: RGB frame array
            frame_idx: Frame index
            timestamp: Time in seconds

        Returns:
            Frame analysis results
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

            frame_result = {
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp,
                "faces_detected": len(response.faces),
                "analysis_timestamp": datetime.now().isoformat(),
                "faces": [],
            }

            # Process each detected face
            for face_idx, face in enumerate(response.faces):
                face_data = self._extract_comprehensive_face_data(face, face_idx)
                frame_result["faces"].append(face_data)

            return frame_result

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_idx}: {e}")
            return {
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp,
                "error": str(e),
                "faces_detected": 0,
                "faces": [],
            }

    def _extract_comprehensive_face_data(self, face, face_idx: int) -> Dict:
        """Extract all available data from detected face."""
        face_data = {
            "face_id": face_idx,
            "detection_confidence": (
                float(face.score) if hasattr(face, "score") else None
            ),
        }

        # Bounding box
        if hasattr(face, "loc") and face.loc is not None:
            face_data["bounding_box"] = {
                "x1": float(face.loc[0]),
                "y1": float(face.loc[1]),
                "x2": float(face.loc[2]),
                "y2": float(face.loc[3]),
                "width": float(face.loc[2] - face.loc[0]),
                "height": float(face.loc[3] - face.loc[1]),
            }

        # Process predictions
        if hasattr(face, "preds") and face.preds:

            # Action Units Analysis (Primary feature)
            if "au" in face.preds:
                face_data["action_units"] = self._process_action_units(face.preds["au"])

            # Facial Expression Recognition
            if "fer" in face.preds:
                face_data["expression"] = self._process_expression(face.preds["fer"])

            # Valence-Arousal Analysis
            if "va" in face.preds:
                face_data["valence_arousal"] = self._process_valence_arousal(
                    face.preds["va"]
                )

            # Face Embeddings
            if "embed" in face.preds:
                embed_pred = face.preds["embed"]
                if hasattr(embed_pred, "logits"):
                    # Store only first 10 dimensions to save space
                    embeddings = embed_pred.logits.cpu().numpy().tolist()
                    face_data["face_embedding_sample"] = (
                        embeddings[:10] if len(embeddings) > 10 else embeddings
                    )

        # Facial landmarks
        if hasattr(face, "landmarks") and face.landmarks is not None:
            face_data["landmarks_count"] = len(face.landmarks)
            # Store key landmarks only (eyes, nose, mouth corners)
            if len(face.landmarks) >= 68:  # Standard 68-point landmarks
                key_points = {
                    "left_eye_center": face.landmarks[36:42].mean(axis=0).tolist(),
                    "right_eye_center": face.landmarks[42:48].mean(axis=0).tolist(),
                    "nose_tip": face.landmarks[30].tolist(),
                    "mouth_left": face.landmarks[48].tolist(),
                    "mouth_right": face.landmarks[54].tolist(),
                    "mouth_center": face.landmarks[62].tolist(),
                }
                face_data["key_landmarks"] = key_points

        return face_data

    def _process_action_units(self, au_pred) -> Dict:
        """Process Action Units predictions with FACS interpretation."""
        au_data = {}

        # Raw predictions
        if hasattr(au_pred, "logits") and au_pred.logits is not None:
            au_data["raw_predictions"] = au_pred.logits.cpu().numpy().tolist()

        # Labels and scores
        if hasattr(au_pred, "labels"):
            au_data["labels"] = au_pred.labels

        if hasattr(au_pred, "scores") and au_pred.scores:
            au_data["scores"] = {k: float(v) for k, v in au_pred.scores.items()}

            # Interpret Action Units with FACS meanings
            au_data["interpretations"] = {}
            active_aus = []

            for au_code, score in au_pred.scores.items():
                score_val = float(score)
                is_active = score_val > 0.5

                au_data["interpretations"][au_code] = {
                    "score": score_val,
                    "active": is_active,
                    "description": self.ACTION_UNITS.get(
                        au_code, "Unknown Action Unit"
                    ),
                    "intensity": self._get_intensity_level(score_val),
                }

                if is_active:
                    active_aus.append(au_code)

            au_data["active_action_units"] = active_aus
            au_data["total_active"] = len(active_aus)

        return au_data

    def _process_expression(self, fer_pred) -> Dict:
        """Process facial expression recognition results."""
        expression_data = {}

        if hasattr(fer_pred, "label"):
            expression_data["predicted_emotion"] = fer_pred.label

        if hasattr(fer_pred, "score"):
            expression_data["confidence"] = float(fer_pred.score)

        if hasattr(fer_pred, "scores") and fer_pred.scores:
            expression_data["all_emotions"] = {
                k: float(v) for k, v in fer_pred.scores.items()
            }

            # Find top 3 emotions
            sorted_emotions = sorted(
                fer_pred.scores.items(), key=lambda x: x[1], reverse=True
            )
            expression_data["top_3_emotions"] = [
                {"emotion": emotion, "score": float(score)}
                for emotion, score in sorted_emotions[:3]
            ]

        return expression_data

    def _process_valence_arousal(self, va_pred) -> Dict:
        """Process valence-arousal predictions."""
        va_data = {}

        if hasattr(va_pred, "valence"):
            va_data["valence"] = float(va_pred.valence)
            va_data["valence_interpretation"] = self._interpret_valence(va_pred.valence)

        if hasattr(va_pred, "arousal"):
            va_data["arousal"] = float(va_pred.arousal)
            va_data["arousal_interpretation"] = self._interpret_arousal(va_pred.arousal)

        # Quadrant analysis
        if "valence" in va_data and "arousal" in va_data:
            va_data["emotional_quadrant"] = self._get_emotional_quadrant(
                va_data["valence"], va_data["arousal"]
            )

        return va_data

    def _get_intensity_level(self, score: float) -> str:
        """Convert AU score to intensity level."""
        if score < 0.3:
            return "None"
        elif score < 0.5:
            return "Trace"
        elif score < 0.7:
            return "Slight"
        elif score < 0.85:
            return "Moderate"
        else:
            return "Strong"

    def _interpret_valence(self, valence: float) -> str:
        """Interpret valence value."""
        if valence < -0.3:
            return "Negative"
        elif valence > 0.3:
            return "Positive"
        else:
            return "Neutral"

    def _interpret_arousal(self, arousal: float) -> str:
        """Interpret arousal value."""
        if arousal < -0.3:
            return "Calm"
        elif arousal > 0.3:
            return "Activated"
        else:
            return "Neutral"

    def _get_emotional_quadrant(self, valence: float, arousal: float) -> str:
        """Determine emotional quadrant based on valence-arousal."""
        if valence > 0 and arousal > 0:
            return "Happy/Excited"
        elif valence > 0 and arousal < 0:
            return "Peaceful/Relaxed"
        elif valence < 0 and arousal > 0:
            return "Angry/Stressed"
        else:
            return "Sad/Depressed"

    def analyze_video(
        self, video_path: str, output_dir: Optional[str] = None, sample_rate: int = 5
    ) -> Dict:
        """
        Analyze complete video for facial actions.

        Args:
            video_path: Path to MP4 video
            output_dir: Output directory for results
            sample_rate: Analyze every nth frame

        Returns:
            Complete analysis results
        """
        logger.info(f"Starting facial action analysis: {video_path}")
        start_time = datetime.now()

        # Extract frames
        frames, video_metadata = self.extract_video_frames(video_path, sample_rate)

        if not frames:
            raise ValueError("No frames extracted from video")

        # Initialize results structure
        results = {
            "analysis_metadata": {
                "video_path": str(video_path),
                "analysis_start_time": start_time.isoformat(),
                "facetorch_device": self.device,
                "sample_rate": sample_rate,
            },
            "video_metadata": video_metadata,
            "frame_analyses": [],
        }

        # Analyze frames
        for idx, frame in enumerate(frames):
            if idx % 50 == 0:
                logger.info(f"Processing frame {idx + 1}/{len(frames)}")

            timestamp = (
                (idx * sample_rate) / video_metadata["fps"]
                if video_metadata["fps"] > 0
                else 0
            )
            frame_result = self.analyze_frame(frame, idx, timestamp)
            results["frame_analyses"].append(frame_result)

        # Add completion metadata
        end_time = datetime.now()
        results["analysis_metadata"]["analysis_end_time"] = end_time.isoformat()
        results["analysis_metadata"]["processing_duration_seconds"] = (
            end_time - start_time
        ).total_seconds()
        results["analysis_metadata"]["frames_processed"] = len(frames)

        # Generate summary statistics
        results["summary_statistics"] = self._generate_summary_stats(results)

        # Save results
        if output_dir:
            self._save_comprehensive_results(results, output_dir)

        logger.info(
            f"Analysis completed in {results['analysis_metadata']['processing_duration_seconds']:.1f}s"
        )
        return results

    def _generate_summary_stats(self, results: Dict) -> Dict:
        """Generate summary statistics from analysis results."""
        stats = {
            "total_frames": len(results["frame_analyses"]),
            "frames_with_faces": 0,
            "total_faces_detected": 0,
            "average_faces_per_frame": 0,
            "most_common_emotion": None,
            "average_valence": None,
            "average_arousal": None,
            "most_active_action_units": [],
        }

        all_emotions = []
        all_valences = []
        all_arousals = []
        au_counts = {}

        for frame in results["frame_analyses"]:
            if frame["faces_detected"] > 0:
                stats["frames_with_faces"] += 1
                stats["total_faces_detected"] += frame["faces_detected"]

                for face in frame["faces"]:
                    # Collect emotions
                    if (
                        "expression" in face
                        and "predicted_emotion" in face["expression"]
                    ):
                        all_emotions.append(face["expression"]["predicted_emotion"])

                    # Collect valence-arousal
                    if "valence_arousal" in face:
                        if "valence" in face["valence_arousal"]:
                            all_valences.append(face["valence_arousal"]["valence"])
                        if "arousal" in face["valence_arousal"]:
                            all_arousals.append(face["valence_arousal"]["arousal"])

                    # Count active action units
                    if (
                        "action_units" in face
                        and "active_action_units" in face["action_units"]
                    ):
                        for au in face["action_units"]["active_action_units"]:
                            au_counts[au] = au_counts.get(au, 0) + 1

        # Calculate averages and most common values
        if stats["frames_with_faces"] > 0:
            stats["average_faces_per_frame"] = (
                stats["total_faces_detected"] / stats["frames_with_faces"]
            )

        if all_emotions:
            stats["most_common_emotion"] = max(
                set(all_emotions), key=all_emotions.count
            )

        if all_valences:
            stats["average_valence"] = sum(all_valences) / len(all_valences)

        if all_arousals:
            stats["average_arousal"] = sum(all_arousals) / len(all_arousals)

        if au_counts:
            stats["most_active_action_units"] = sorted(
                au_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

        return stats

    def _save_comprehensive_results(self, results: Dict, output_dir: str):
        """Save results in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete JSON results
        json_file = output_path / f"facial_analysis_complete_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Complete results: {json_file}")

        # Save summary CSV
        summary_csv = output_path / f"facial_analysis_summary_{timestamp}.csv"
        self._create_summary_csv(results, summary_csv)
        logger.info(f"Summary CSV: {summary_csv}")

        # Save Action Units timeline CSV
        au_csv = output_path / f"action_units_timeline_{timestamp}.csv"
        self._create_action_units_csv(results, au_csv)
        logger.info(f"Action Units CSV: {au_csv}")

        # Save analysis report
        report_file = output_path / f"analysis_report_{timestamp}.txt"
        self._create_analysis_report(results, report_file)
        logger.info(f"Analysis report: {report_file}")

    def _create_summary_csv(self, results: Dict, csv_path: Path):
        """Create summary CSV with key metrics per frame."""
        rows = []

        for frame in results["frame_analyses"]:
            base_row = {
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame.get("timestamp_seconds", 0),
                "faces_detected": frame["faces_detected"],
            }

            if frame.get("faces"):
                for face in frame["faces"]:
                    row = base_row.copy()
                    row["face_id"] = face["face_id"]
                    row["detection_confidence"] = face.get("detection_confidence")

                    # Expression
                    if "expression" in face:
                        row["emotion"] = face["expression"].get("predicted_emotion")
                        row["emotion_confidence"] = face["expression"].get("confidence")

                    # Valence-Arousal
                    if "valence_arousal" in face:
                        row["valence"] = face["valence_arousal"].get("valence")
                        row["arousal"] = face["valence_arousal"].get("arousal")
                        row["emotional_quadrant"] = face["valence_arousal"].get(
                            "emotional_quadrant"
                        )

                    # Action Units summary
                    if "action_units" in face:
                        row["active_aus_count"] = face["action_units"].get(
                            "total_active", 0
                        )
                        row["active_aus"] = ",".join(
                            face["action_units"].get("active_action_units", [])
                        )

                    rows.append(row)
            else:
                rows.append(base_row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

    def _create_action_units_csv(self, results: Dict, csv_path: Path):
        """Create detailed Action Units timeline."""
        rows = []

        for frame in results["frame_analyses"]:
            for face in frame.get("faces", []):
                if "action_units" in face and "interpretations" in face["action_units"]:
                    for au_code, au_info in face["action_units"][
                        "interpretations"
                    ].items():
                        rows.append(
                            {
                                "frame_index": frame["frame_index"],
                                "timestamp_seconds": frame.get("timestamp_seconds", 0),
                                "face_id": face["face_id"],
                                "action_unit": au_code,
                                "description": au_info["description"],
                                "score": au_info["score"],
                                "active": au_info["active"],
                                "intensity": au_info["intensity"],
                            }
                        )

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

    def _create_analysis_report(self, results: Dict, report_path: Path):
        """Create human-readable analysis report."""
        with open(report_path, "w") as f:
            f.write("FACIAL ACTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Video information
            f.write("VIDEO INFORMATION:\n")
            f.write(f"File: {results['analysis_metadata']['video_path']}\n")
            f.write(
                f"Duration: {results['video_metadata'].get('duration_seconds', 0):.1f} seconds\n"
            )
            f.write(f"Total Frames: {results['video_metadata']['total_frames']}\n")
            f.write(f"FPS: {results['video_metadata']['fps']:.1f}\n")
            f.write(
                f"Resolution: {results['video_metadata']['width']}x{results['video_metadata']['height']}\n\n"
            )

            # Analysis summary
            f.write("ANALYSIS SUMMARY:\n")
            stats = results["summary_statistics"]
            f.write(f"Frames Analyzed: {stats['total_frames']}\n")
            f.write(f"Frames with Faces: {stats['frames_with_faces']}\n")
            f.write(f"Total Faces Detected: {stats['total_faces_detected']}\n")
            f.write(
                f"Average Faces per Frame: {stats['average_faces_per_frame']:.2f}\n"
            )

            if stats["most_common_emotion"]:
                f.write(f"Most Common Emotion: {stats['most_common_emotion']}\n")

            if stats["average_valence"] is not None:
                f.write(f"Average Valence: {stats['average_valence']:.3f}\n")

            if stats["average_arousal"] is not None:
                f.write(f"Average Arousal: {stats['average_arousal']:.3f}\n")

            if stats["most_active_action_units"]:
                f.write("\nMOST ACTIVE ACTION UNITS:\n")
                for au, count in stats["most_active_action_units"]:
                    description = self.ACTION_UNITS.get(au, "Unknown")
                    f.write(f"{au} ({description}): {count} occurrences\n")

            f.write(
                f"\nAnalysis completed in {results['analysis_metadata']['processing_duration_seconds']:.1f} seconds\n"
            )


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze MP4 video for facial actions using facetorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_facetorch_analyzer.py video.mp4 --output results/
  python enhanced_facetorch_analyzer.py video.mp4 --sample-rate 10 --output results/
        """,
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
    parser.add_argument("--config", "-c", help="Path to custom facetorch config file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        analyzer = FacialActionVideoAnalyzer(config_path=args.config)
        results = analyzer.analyze_video(
            video_path=args.video_path,
            output_dir=args.output,
            sample_rate=args.sample_rate,
        )

        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Frames analyzed: {results['summary_statistics']['total_frames']}")
        print(
            f"👥 Faces detected: {results['summary_statistics']['total_faces_detected']}"
        )
        print(
            f"⏱️  Processing time: {results['analysis_metadata']['processing_duration_seconds']:.1f}s"
        )

        if results["summary_statistics"]["most_common_emotion"]:
            print(
                f"😊 Most common emotion: {results['summary_statistics']['most_common_emotion']}"
            )

        if args.output:
            print(f"💾 Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
