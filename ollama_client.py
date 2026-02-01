"""
Ollama Integration Module for AI Video Editor
Handles communication with local Ollama models for clip detection and analysis
Uses direct HTTP requests for stability across Ollama versions
"""
import json
import re
import requests
from typing import Optional, List, Dict
from config import OLLAMA_MODEL, OLLAMA_HOST, SHORT_MIN_DURATION, SHORT_MAX_DURATION


class OllamaClipDetector:
    """Uses Ollama to analyze transcripts and identify engaging clip moments"""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.host = OLLAMA_HOST.rstrip('/')

    def check_connection(self) -> bool:
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [m.get('name', '') for m in models]
                # Check if our model exists (with or without :latest tag)
                base_model = self.model.split(':')[0]
                for name in model_names:
                    if base_model in name or name.startswith(base_model):
                        return True
                print(f"Model '{self.model}' not found. Available models: {model_names}")
                return False
            return False
        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama. Is it running?")
            return False
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return False

    def _chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Send chat request to Ollama API"""
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 2048
                    }
                },
                timeout=180  # 3 minutes for long transcripts
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {}).get('content', '')
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return ''
        except requests.exceptions.Timeout:
            print("Ollama request timed out. The model may be processing a large transcript.")
            return ''
        except Exception as e:
            print(f"Ollama chat error: {e}")
            return ''

    def analyze_transcript_for_clips(
        self,
        transcript: List[Dict],
        num_clips: int = 5,
        context: Optional[str] = None
    ) -> List[Dict]:
        """
        Analyze transcript and identify the most engaging moments for short clips.

        Args:
            transcript: List of dicts with 'start', 'end', 'text' keys
            num_clips: Number of clips to identify
            context: Optional context about the video content

        Returns:
            List of clip suggestions with start/end times and reasoning
        """
        # Format transcript for the model
        formatted_transcript = self._format_transcript(transcript)

        prompt = self._build_analysis_prompt(
            formatted_transcript,
            num_clips,
            context
        )

        messages = [
            {
                "role": "system",
                "content": """You are an expert video editor specializing in creating viral short-form content.
Your task is to analyze video transcripts and identify the most engaging moments that would make great short clips.
Look for: hooks, emotional moments, key insights, funny parts, dramatic statements, surprising reveals, or compelling stories.
Always respond with valid JSON only. No explanations, just the JSON array."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print("Sending transcript to AI for analysis...")
        response = self._chat(messages)

        if response:
            clips = self._parse_clip_response(response)
            print(f"AI identified {len(clips)} potential clips")
            return clips

        print("No response from AI")
        return []

    def generate_clip_title(self, clip_text: str) -> str:
        """Generate a catchy title for a clip"""
        messages = [
            {
                "role": "system",
                "content": "Generate a short, catchy title (max 10 words) for this video clip. Return only the title, nothing else."
            },
            {
                "role": "user",
                "content": f"Clip content: {clip_text[:500]}"
            }
        ]

        response = self._chat(messages, temperature=0.8)
        if response:
            return response.strip().strip('"\'')
        return "Untitled Clip"

    def _format_transcript(self, transcript: List[Dict]) -> str:
        """Format transcript with timestamps for model analysis"""
        lines = []
        for segment in transcript:
            start = self._format_time(segment['start'])
            end = self._format_time(segment['end'])
            text = segment['text'].strip()
            lines.append(f"[{start} - {end}] {text}")
        return "\n".join(lines)

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _parse_time(self, time_str: str) -> float:
        """Convert MM:SS or HH:MM:SS to seconds"""
        try:
            parts = time_str.strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except (ValueError, AttributeError):
            pass
        return 0.0

    def _build_analysis_prompt(
        self,
        transcript: str,
        num_clips: int,
        context: Optional[str]
    ) -> str:
        """Build the prompt for clip analysis"""
        context_section = f"\nVideo Context: {context}\n" if context else ""

        return f"""Analyze this video transcript and identify the {num_clips} most engaging moments for short-form clips.
{context_section}
Requirements:
- Each clip should be between {SHORT_MIN_DURATION} and {SHORT_MAX_DURATION} seconds long
- Look for: hooks, emotional peaks, key insights, humor, drama, or compelling narratives
- Clips should be self-contained and make sense without additional context
- Prioritize moments that would capture attention in the first 3 seconds

Transcript:
{transcript}

Respond with a JSON array of clips in this exact format:
[
  {{
    "start_time": "MM:SS",
    "end_time": "MM:SS",
    "reason": "Brief explanation of why this moment is engaging",
    "hook": "Suggested hook or caption for the clip",
    "virality_score": 8
  }}
]

Return ONLY the JSON array, no other text."""

    def _parse_clip_response(self, response: str) -> List[Dict]:
        """Parse the model's response into clip data"""
        try:
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'^```(?:json)?\n?', '', response)
                response = re.sub(r'\n?```$', '', response)

            # Try to extract JSON from the response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                clips_data = json.loads(json_match.group())
            else:
                clips_data = json.loads(response)

            clips = []
            for clip in clips_data:
                start = self._parse_time(clip.get('start_time', '0:00'))
                end = self._parse_time(clip.get('end_time', '0:30'))

                # Ensure minimum duration
                if end - start < SHORT_MIN_DURATION:
                    end = start + SHORT_MIN_DURATION

                clips.append({
                    'start': start,
                    'end': end,
                    'reason': clip.get('reason', ''),
                    'hook': clip.get('hook', ''),
                    'score': clip.get('virality_score', 5)
                })

            # Sort by virality score
            clips.sort(key=lambda x: x['score'], reverse=True)
            return clips

        except json.JSONDecodeError as e:
            print(f"Failed to parse clip response as JSON: {e}")
            print(f"Raw response (first 500 chars): {response[:500]}")
            return []
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []


def test_ollama_connection():
    """Test function to verify Ollama setup"""
    detector = OllamaClipDetector()
    if detector.check_connection():
        print(f"Connected to Ollama with model: {OLLAMA_MODEL}")
        return True
    else:
        print(f"Could not connect to Ollama or model '{OLLAMA_MODEL}' not found")
        print("Make sure Ollama is running: ollama serve")
        print(f"And the model is pulled: ollama pull {OLLAMA_MODEL}")
        return False


if __name__ == "__main__":
    test_ollama_connection()
