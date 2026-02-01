"""
Video Processing Module for AI Video Editor
Handles video loading, transcription, clip extraction, and format conversion
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Callable, List, Dict
import whisper
import torch
import cv2
import numpy as np

from config import (
    WHISPER_MODEL, TEMP_DIR, OUTPUT_DIR,
    SHORT_RESOLUTION, SHORT_ASPECT_RATIO,
    SHORT_MIN_DURATION, SHORT_MAX_DURATION
)

# Delayed import to avoid numpy/opencv issues at module load
_moviepy_imported = False
VideoFileClip = None
crop = None
resize = None


def _import_moviepy():
    """Lazy import moviepy to avoid startup issues"""
    global _moviepy_imported, VideoFileClip, crop, resize
    if not _moviepy_imported:
        from moviepy.editor import VideoFileClip as VFC
        from moviepy.video.fx.all import crop as _crop, resize as _resize
        VideoFileClip = VFC
        crop = _crop
        resize = _resize
        _moviepy_imported = True


class FaceDetector:
    """Detects faces in video frames to enable smart cropping"""

    def __init__(self):
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_face_center(self, frame) -> Optional[tuple]:
        """
        Detect face in frame and return center coordinates.
        Returns (x_center, y_center) or None if no face found.
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) > 0:
            # Get the largest face (most likely the main speaker)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y)

        return None

    def get_average_face_position(self, clip, num_samples: int = 10) -> Optional[tuple]:
        """
        Sample multiple frames from the clip to get average face position.
        This provides a stable crop position throughout the clip.
        """
        duration = clip.duration
        face_positions = []

        # Sample frames throughout the clip
        for i in range(num_samples):
            try:
                t = (i / num_samples) * duration
                frame = clip.get_frame(t)
                face_center = self.detect_face_center(frame)
                if face_center:
                    face_positions.append(face_center)
            except Exception:
                continue

        if face_positions:
            # Return average position
            avg_x = int(sum(p[0] for p in face_positions) / len(face_positions))
            avg_y = int(sum(p[1] for p in face_positions) / len(face_positions))
            return (avg_x, avg_y)

        return None


class VideoTranscriber:
    """Handles audio transcription with word-level timestamps using Whisper"""

    def __init__(self, model_size: str = WHISPER_MODEL):
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, progress_callback: Optional[Callable] = None):
        """Load Whisper model (lazy loading)"""
        if self.model is None:
            if progress_callback:
                progress_callback("Loading Whisper model...")
            print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            if progress_callback:
                progress_callback(f"Whisper model loaded on {self.device}")
            print("Whisper model loaded successfully")

    def transcribe(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Transcribe video audio with word-level timestamps.

        Returns:
            dict with 'segments' (sentence-level) and 'words' (word-level) data
        """
        _import_moviepy()
        self.load_model(progress_callback)

        if progress_callback:
            progress_callback("Extracting audio from video...")

        print("Extracting audio from video...")

        # Extract audio to temp file
        audio_path = self._extract_audio(video_path)

        try:
            if progress_callback:
                progress_callback("Transcribing audio (this may take a while)...")

            print("Transcribing audio...")

            # Transcribe with word timestamps
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )

            # Process results
            segments = []
            all_words = []

            for segment in result['segments']:
                seg_data = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                }
                segments.append(seg_data)

                # Extract word-level data
                if 'words' in segment:
                    for word in segment['words']:
                        all_words.append({
                            'word': word['word'].strip(),
                            'start': word['start'],
                            'end': word['end']
                        })

            if progress_callback:
                progress_callback(f"Transcription complete: {len(segments)} segments, {len(all_words)} words")

            print(f"Transcription complete: {len(segments)} segments, {len(all_words)} words")

            return {
                'segments': segments,
                'words': all_words,
                'text': result['text'],
                'language': result.get('language', 'en')
            }

        finally:
            # Cleanup temp audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        os.makedirs(TEMP_DIR, exist_ok=True)
        audio_path = os.path.join(TEMP_DIR, "temp_audio.wav")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return audio_path


class VideoClipper:
    """Handles video clip extraction and formatting"""

    def __init__(self):
        self.target_width, self.target_height = SHORT_RESOLUTION
        self.face_detector = FaceDetector()

    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None,
        convert_vertical: bool = True,
        focus_on_face: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Extract a clip from video and optionally convert to vertical format.

        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path (auto-generated if None)
            convert_vertical: Whether to convert to 9:16 aspect ratio
            focus_on_face: Whether to detect and focus on speaker's face
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the extracted clip
        """
        _import_moviepy()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if output_path is None:
            filename = f"clip_{int(start_time)}_{int(end_time)}.mp4"
            output_path = os.path.join(OUTPUT_DIR, filename)

        if progress_callback:
            progress_callback(f"Extracting clip: {start_time:.1f}s - {end_time:.1f}s")

        print(f"Extracting clip: {start_time:.1f}s - {end_time:.1f}s")

        # Load video and extract subclip
        video = VideoFileClip(video_path)
        clip = video.subclip(start_time, min(end_time, video.duration))

        if convert_vertical:
            clip = self._convert_to_vertical(clip, focus_on_face=focus_on_face)

        # Write output
        if progress_callback:
            progress_callback("Encoding clip...")

        print(f"Encoding clip to {output_path}...")

        clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )

        clip.close()
        video.close()

        if progress_callback:
            progress_callback(f"Clip saved: {output_path}")

        print(f"Clip saved: {output_path}")

        return output_path

    def _convert_to_vertical(self, clip, focus_on_face: bool = True):
        """Convert horizontal video to vertical (9:16) format, focusing on detected face"""
        original_w, original_h = clip.size
        target_w, target_h = self.target_width, self.target_height
        target_ratio = target_w / target_h

        # Calculate current aspect ratio
        current_ratio = original_w / original_h

        # Default to center crop
        x_center = original_w // 2
        y_center = original_h // 2

        # Try to detect face for smart cropping
        if focus_on_face:
            print("Detecting face position for smart cropping...")
            face_pos = self.face_detector.get_average_face_position(clip, num_samples=15)
            if face_pos:
                x_center, y_center = face_pos
                print(f"Face detected at position: ({x_center}, {y_center})")
            else:
                print("No face detected, using center crop")

        if current_ratio > target_ratio:
            # Video is wider than target - crop sides
            new_width = int(original_h * target_ratio)

            # Calculate x1 based on face position (or center)
            x1 = x_center - new_width // 2

            # Ensure we don't go out of bounds
            x1 = max(0, min(x1, original_w - new_width))

            clip = crop(clip, x1=x1, width=new_width)
        else:
            # Video is taller/same as target - crop top/bottom
            new_height = int(original_w / target_ratio)

            # Calculate y1 based on face position (or center)
            y1 = y_center - new_height // 2

            # Ensure we don't go out of bounds
            y1 = max(0, min(y1, original_h - new_height))

            clip = crop(clip, y1=y1, height=new_height)

        # Resize to target resolution
        clip = resize(clip, newsize=(target_w, target_h))

        return clip

    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata"""
        _import_moviepy()
        video = VideoFileClip(video_path)
        info = {
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,
            'width': video.size[0],
            'height': video.size[1],
            'has_audio': video.audio is not None
        }
        video.close()
        return info


class ClipProcessor:
    """Main class that orchestrates the video processing pipeline"""

    def __init__(self):
        self.transcriber = VideoTranscriber()
        self.clipper = VideoClipper()

    def process_video(
        self,
        video_path: str,
        clips_data: List[Dict],
        add_captions: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Process video to create multiple clips.

        Args:
            video_path: Path to source video
            clips_data: List of clip definitions with start/end times
            add_captions: Whether to add captions (handled separately)
            progress_callback: Optional callback for progress updates

        Returns:
            List of paths to created clips
        """
        output_paths = []

        for i, clip_info in enumerate(clips_data):
            if progress_callback:
                progress_callback(f"Processing clip {i + 1}/{len(clips_data)}")

            start = clip_info['start']
            end = clip_info['end']

            # Validate clip duration
            duration = end - start
            if duration < SHORT_MIN_DURATION:
                end = start + SHORT_MIN_DURATION
            elif duration > SHORT_MAX_DURATION:
                end = start + SHORT_MAX_DURATION

            output_path = self.clipper.extract_clip(
                video_path=video_path,
                start_time=start,
                end_time=end,
                progress_callback=progress_callback
            )
            output_paths.append(output_path)

        return output_paths

    def get_words_for_timerange(
        self,
        words: List[Dict],
        start_time: float,
        end_time: float
    ) -> List[Dict]:
        """Get words that fall within a specific time range, adjusted to clip time"""
        clip_words = []
        for word in words:
            if word['start'] >= start_time and word['end'] <= end_time:
                clip_words.append({
                    'word': word['word'],
                    'start': word['start'] - start_time,  # Adjust to clip time
                    'end': word['end'] - start_time
                })
        return clip_words


def test_video_processing():
    """Test function for video processing"""
    print("Video processing module loaded successfully")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Whisper model: {WHISPER_MODEL}")


if __name__ == "__main__":
    test_video_processing()
