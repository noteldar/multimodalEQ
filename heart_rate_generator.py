import json
import numpy as np
import matplotlib.pyplot as plt


def process_vitallens_json(json_file_path):
    """
    Process VitalLens JSON to generate heart rate array over time

    Args:
        json_file_path (str): Path to the JSON file

    Returns:
        dict: Contains time array and heart rate array
    """

    # Load JSON data
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # Handle list format (take first element if it's a list)
    if isinstance(data, list):
        data = data[0]

    # Extract vital signs
    vital_signs = data["vital_signs"]

    # Get average heart rate
    avg_heart_rate = vital_signs["heart_rate"]["value"]
    hr_confidence = vital_signs["heart_rate"]["confidence"]

    # Get PPG waveform data
    ppg_waveform = vital_signs["ppg_waveform"]["data"]
    ppg_confidence = vital_signs["ppg_waveform"]["confidence"]

    # Create time array (assuming 30 FPS sampling rate)
    sample_rate = 30.0  # frames per second
    num_samples = len(ppg_waveform)
    time_array = np.arange(num_samples) / sample_rate

    # Normalize PPG waveform to create heart rate variations
    ppg_array = np.array(ppg_waveform)
    ppg_normalized = ppg_array - np.mean(ppg_array)  # Remove DC component

    # Scale PPG variations to reasonable heart rate range (±10% of average)
    variation_factor = 0.1
    if np.std(ppg_normalized) > 0:
        ppg_normalized = ppg_normalized / np.std(ppg_normalized)

    # Apply confidence weighting
    confidence_array = np.array(ppg_confidence)
    weighted_ppg = ppg_normalized * confidence_array

    # Generate heart rate variations around the average
    hr_variation = weighted_ppg * avg_heart_rate * variation_factor
    heart_rate_array = avg_heart_rate + hr_variation

    # Ensure heart rate stays within physiological bounds
    heart_rate_array = np.clip(heart_rate_array, 30, 200)

    # Compile results
    results = {
        "time_seconds": time_array,
        "heart_rate_bpm": heart_rate_array,
        "average_heart_rate": avg_heart_rate,
        "hr_confidence": hr_confidence,
        "duration_seconds": time_array[-1],
        "num_samples": num_samples,
        "sample_rate": sample_rate,
    }

    return results


def plot_heart_rate(results):
    """Plot heart rate over time"""

    time_array = results["time_seconds"]
    hr_array = results["heart_rate_bpm"]
    avg_hr = results["average_heart_rate"]

    plt.figure(figsize=(12, 6))
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
    plt.tight_layout()
    plt.show()


def save_to_csv(results, filename="heart_rate_data.csv"):
    """Save heart rate data to CSV file"""

    import pandas as pd

    df = pd.DataFrame(
        {
            "time_seconds": results["time_seconds"],
            "heart_rate_bpm": results["heart_rate_bpm"],
        }
    )

    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Process the JSON file
    json_file = "vitallens_khabib.json"
    json_file = "vitallens_henry.json"
    json_file = "vitallens_annoyed.json"

    try:
        # Generate heart rate data
        results = process_vitallens_json(json_file)

        # Print summary
        print("Heart Rate Analysis Results:")
        print("=" * 40)
        print(f"Average Heart Rate: {results['average_heart_rate']:.1f} BPM")
        print(f"Confidence: {results['hr_confidence']:.3f}")
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Number of samples: {results['num_samples']}")
        print(f"Sample rate: {results['sample_rate']} FPS")
        print(
            f"HR Range: {np.min(results['heart_rate_bpm']):.1f} - {np.max(results['heart_rate_bpm']):.1f} BPM"
        )

        # Plot the results
        plot_heart_rate(results)

        # Save to CSV (optional)
        save_to_csv(results)

        # Access the heart rate array
        heart_rate_bpm = results["heart_rate_bpm"]
        time_seconds = results["time_seconds"]

        print(f"\nFirst 10 heart rate values:")
        for i in range(min(10, len(heart_rate_bpm))):
            print(f"Time: {time_seconds[i]:.2f}s, HR: {heart_rate_bpm[i]:.1f} BPM")

    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file}'")
        print("Please make sure the JSON file is in the current directory.")
    except Exception as e:
        print(f"Error processing data: {e}")
