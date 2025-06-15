#!/usr/bin/env python3
"""
VitalLens Heart Rate Analyzer
Generates heart rate values over time from VitalLens JSON data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


def load_vitallens_data(json_file_path: str) -> Dict[Any, Any]:
    """
    Load VitalLens JSON data from file

    Args:
        json_file_path: Path to the JSON file

    Returns:
        Parsed JSON data
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data[0] if isinstance(data, list) else data


def generate_heart_rate_over_time(
    avg_heart_rate: float,
    ppg_waveform: List[float],
    ppg_confidence: List[float],
    sample_rate_fps: float = 30.0,
    variation_factor: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heart rate values over time based on average HR and PPG waveform

    Args:
        avg_heart_rate: Average heart rate in BPM
        ppg_waveform: PPG waveform data (unitless)
        ppg_confidence: Confidence values for each PPG sample
        sample_rate_fps: Sampling rate in frames per second
        variation_factor: Factor controlling HR variation around average (0.1 = ±10% variation)

    Returns:
        Tuple of (time_array, heart_rate_array)
    """
    # Create time array based on sample rate
    num_samples = len(ppg_waveform)
    time_array = np.arange(num_samples) / sample_rate_fps

    # Normalize PPG waveform to create variation around average HR
    ppg_normalized = np.array(ppg_waveform)
    ppg_normalized = ppg_normalized - np.mean(ppg_normalized)  # Remove DC component
    ppg_std = np.std(ppg_normalized)

    if ppg_std > 0:
        ppg_normalized = ppg_normalized / ppg_std  # Normalize to unit variance

    # Apply confidence weighting - lower confidence reduces variation
    confidence_array = np.array(ppg_confidence)
    weighted_variation = ppg_normalized * confidence_array

    # Generate heart rate variations around the average
    hr_variation = weighted_variation * avg_heart_rate * variation_factor
    heart_rate_array = avg_heart_rate + hr_variation

    # Ensure heart rate stays within physiological bounds (30-200 BPM)
    heart_rate_array = np.clip(heart_rate_array, 30, 200)

    return time_array, heart_rate_array


def analyze_vitallens_json(
    json_file_path: str, output_plot: bool = True
) -> Dict[str, Any]:
    """
    Main function to analyze VitalLens JSON and generate heart rate over time

    Args:
        json_file_path: Path to the VitalLens JSON file
        output_plot: Whether to generate and show a plot

    Returns:
        Dictionary containing analysis results
    """
    # Load data
    data = load_vitallens_data(json_file_path)

    # Extract vital signs data
    vital_signs = data["vital_signs"]

    # Get average heart rate
    avg_hr = vital_signs["heart_rate"]["value"]
    hr_confidence = vital_signs["heart_rate"]["confidence"]

    # Get PPG waveform data
    ppg_data = vital_signs["ppg_waveform"]["data"]
    ppg_confidence = vital_signs["ppg_waveform"]["confidence"]

    # Generate heart rate over time
    time_array, hr_array = generate_heart_rate_over_time(
        avg_hr, ppg_data, ppg_confidence
    )

    # Calculate statistics
    results = {
        "average_heart_rate": avg_hr,
        "hr_confidence": hr_confidence,
        "time_seconds": time_array,
        "heart_rate_bpm": hr_array,
        "duration_seconds": time_array[-1],
        "num_samples": len(hr_array),
        "hr_min": np.min(hr_array),
        "hr_max": np.max(hr_array),
        "hr_std": np.std(hr_array),
        "sample_rate_fps": 30.0,
    }

    # Print summary
    print(f"VitalLens Heart Rate Analysis Summary:")
    print(f"=====================================")
    print(f"Average Heart Rate: {avg_hr:.1f} BPM (confidence: {hr_confidence:.3f})")
    print(f"Duration: {results['duration_seconds']:.1f} seconds")
    print(f"Number of samples: {results['num_samples']}")
    print(f"HR Range: {results['hr_min']:.1f} - {results['hr_max']:.1f} BPM")
    print(f"HR Standard Deviation: {results['hr_std']:.1f} BPM")
    print(f"Sample Rate: {results['sample_rate_fps']} FPS")

    # Generate plot if requested
    if output_plot:
        plt.figure(figsize=(12, 8))

        # Plot heart rate over time
        plt.subplot(2, 1, 1)
        plt.plot(time_array, hr_array, "r-", linewidth=1.5, label="Heart Rate")
        plt.axhline(
            y=avg_hr,
            color="b",
            linestyle="--",
            alpha=0.7,
            label=f"Average ({avg_hr:.1f} BPM)",
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (BPM)")
        plt.title("Heart Rate Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot PPG waveform for reference
        plt.subplot(2, 1, 2)
        plt.plot(
            time_array, ppg_data, "g-", linewidth=1, alpha=0.7, label="PPG Waveform"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("PPG Signal (unitless)")
        plt.title("PPG Waveform (Reference)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return results


def export_heart_rate_data(
    results: Dict[str, Any], output_file: str = "heart_rate_data.csv"
):
    """
    Export heart rate data to CSV file

    Args:
        results: Results dictionary from analyze_vitallens_json
        output_file: Output CSV filename
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "time_seconds": results["time_seconds"],
            "heart_rate_bpm": results["heart_rate_bpm"],
        }
    )

    df.to_csv(output_file, index=False)
    print(f"\nHeart rate data exported to: {output_file}")


# Example usage
if __name__ == "__main__":
    # Example with the provided JSON file
    json_file = "vitallens_khabib.json"

    try:
        # Analyze the VitalLens data
        results = analyze_vitallens_json(json_file, output_plot=True)

        # Export to CSV
        export_heart_rate_data(results)

        # Access the heart rate array
        heart_rate_over_time = results["heart_rate_bpm"]
        time_points = results["time_seconds"]

        print(f"\nFirst 10 heart rate values:")
        for i in range(min(10, len(heart_rate_over_time))):
            print(f"Time: {time_points[i]:.2f}s, HR: {heart_rate_over_time[i]:.1f} BPM")

    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file}'")
        print("Please make sure the JSON file exists in the current directory.")
    except Exception as e:
        print(f"Error processing data: {e}")
