import json

if __name__ == "__main__":
    files = [
        "HumeAI_predictions_khabib.json",
        "HumeAI_predictions_henry.json",
        "HumeAI_predictions_annoyed.json",
    ]
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                predictions = data[0]["results"]["predictions"][0]["models"]["prosody"][
                    "grouped_predictions"
                ][0]["predictions"]

                # Get top 3 emotions for each time moment
                all_emotions = [
                    sorted(p["emotions"], key=lambda x: x["score"], reverse=True)[:3]
                    for p in predictions
                ]

                # Get all unique emotion names from top 3 emotions
                unique_emotions = set()
                for emotions_at_time in all_emotions:
                    for emotion in emotions_at_time:
                        unique_emotions.add(emotion["name"])

                print(f"Processing file: {file}")
                print(f"Number of time moments: {len(predictions)}")
                print(f"Unique emotions found: {len(unique_emotions)}")
                print(f"Unique emotions: {sorted(unique_emotions)}")
                print()

                # Create final dictionary tracking each emotion across time
                final_dict = {}
                for emotion_name in unique_emotions:
                    emotion_scores = []
                    for prediction in predictions:
                        # Find the score for this emotion at this time moment
                        emotion_score = 0  # default if not found in this time moment
                        for emotion in prediction["emotions"]:
                            if emotion["name"] == emotion_name:
                                emotion_score = emotion["score"]
                                break
                        emotion_scores.append(
                            round(emotion_score, 4)
                        )  # Round for readability
                    final_dict[emotion_name] = emotion_scores

                print("Final emotion tracking dictionary:")
                for emotion, scores in sorted(final_dict.items()):
                    print(f'  "{emotion}": {scores}')

                # Save to JSON file
                output_filename = file.replace(".json", "_emotion_timeline.json")
                with open(output_filename, "w") as out_file:
                    json.dump(final_dict, out_file, indent=2)
                print(f"\nSaved emotion timeline to: {output_filename}")
                print("=" * 50)

        except FileNotFoundError:
            print(f"File {file} not found, skipping...")
            print("=" * 50)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            print("=" * 50)
