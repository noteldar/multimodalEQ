import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Try to import whisper functionality, with fallback
try:
    from transcript import get_video_transcript
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Whisper not available ({e}). Using mock transcript for testing.")
    WHISPER_AVAILABLE = False
    
    def get_video_transcript(video_path: str, model_size: str = "large") -> str:
        """Mock transcript function for testing when whisper is not available"""
        return """Hello everyone, this is a test video. I'm feeling quite excited about this project. 
        There are some challenges ahead but I'm confident we can overcome them. 
        Sometimes I feel a bit anxious about the deadlines, but overall I'm happy with the progress we're making. 
        Thank you for watching and I hope you found this helpful."""

class TextEmotionRecognizer:
    def __init__(self):
        self.api_key = os.getenv('INFLECTION_API_KEY')
        if not self.api_key:
            raise ValueError("INFLECTION_API_KEY not found in environment variables")
        
        self.base_url = "https://api.inflection.ai/v1/chat/completions"
        
    def analyze_video_emotions(self, video_path: str, model_size: str = "large") -> dict:
        """
        Analyze emotions from a video file using transcript and Inflection AI.
        
        Args:
            video_path (str): Path to the video file
            model_size (str): Whisper model size for transcription
            
        Returns:
            dict: JSON response with top 3 emotions and intensity levels
        """
        try:
            # Check if video file exists (skip check for mock)
            if WHISPER_AVAILABLE and not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get transcript from video
            if WHISPER_AVAILABLE:
                print(f"Getting transcript from video: {video_path}")
            else:
                print(f"Using mock transcript (Whisper not available)")
                
            transcript = get_video_transcript(video_path, model_size)
            
            if not transcript.strip():
                raise ValueError("No transcript could be generated from the video")
            
            # Print the transcript
            print("\n--- TRANSCRIPT ---")
            print(transcript)
            print("--- END TRANSCRIPT ---\n")
            
            # Analyze emotions using Inflection AI
            print("Analyzing emotions using Inflection AI...")
            emotions_result = self._analyze_text_emotions(transcript)
            
            # Format the response
            video_filename = Path(video_path).name
            result = {
                "file": video_filename,
                "text_emotions": emotions_result
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Error analyzing video emotions: {str(e)}")
    
    def _analyze_text_emotions(self, text: str) -> list:
        """
        Send text to Inflection AI for emotion analysis.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: List of top 3 emotions with intensity levels
        """
        prompt = """Imagine you are the best emotion recognizer in the world. You have to observe the text and do a complex sentiment analysis and return the top 3 emotions with intensity levels (ranging from 0 (no expression of that emotion) to 10 (highly intense)).

Please analyze the following text and respond with ONLY a JSON array in this exact format:
[
    {"emotion": "emotion_name", "level": intensity_score},
    {"emotion": "emotion_name", "level": intensity_score},
    {"emotion": "emotion_name", "level": intensity_score}
]

Text to analyze:
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Pi-3.1",
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{text}"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            ai_response = response_data['choices'][0]['message']['content'].strip()
            
            # Parse the JSON response
            try:
                emotions_list = json.loads(ai_response)
                
                # Validate the response format
                if not isinstance(emotions_list, list) or len(emotions_list) != 3:
                    raise ValueError("Invalid response format from AI")
                
                for emotion in emotions_list:
                    if not isinstance(emotion, dict) or 'emotion' not in emotion or 'level' not in emotion:
                        raise ValueError("Invalid emotion format in response")
                    
                    # Ensure level is within valid range
                    emotion['level'] = max(0, min(10, int(emotion['level'])))
                
                return emotions_list
                
            except json.JSONDecodeError:
                raise ValueError("Failed to parse AI response as JSON")
                
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def analyze_text_emotions(self, text: str) -> dict:
        """
        Analyze emotions from raw text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: JSON response with top 3 emotions and intensity levels
        """
        emotions_result = self._analyze_text_emotions(text)
        
        result = {
            "file": "text_input",
            "text_emotions": emotions_result
        }
        
        return result
    
    def save_results_to_file(self, results: dict, output_path: str = None) -> str:
        """
        Save emotion analysis results to a JSON file.
        
        Args:
            results (dict): The emotion analysis results
            output_path (str): Optional custom output path
            
        Returns:
            str: Path to the saved file
        """
        if output_path is None:
            video_name = Path(results["file"]).stem
            output_path = f"{video_name}_emotions.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return output_path


def main():
    """Example usage of the TextEmotionRecognizer"""
    import sys
    
    # Use henry.mp4 as the default video file
    video_path = "henry.mp4"
    output_path = None
    
    # Still allow command line override if needed
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Initialize the emotion recognizer
        recognizer = TextEmotionRecognizer()
        
        # Analyze emotions from video
        results = recognizer.analyze_video_emotions(video_path)
        
        # Print results
        print("\n--- EMOTION ANALYSIS RESULTS ---")
        print(json.dumps(results, indent=2))
        
        # Save to file
        if output_path or True:  # Always save by default
            saved_path = recognizer.save_results_to_file(results, output_path)
            print(f"\nResults saved to: {saved_path}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
