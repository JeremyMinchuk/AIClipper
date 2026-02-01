"""
Caption Generator Module for AI Video Editor
Creates animated word-by-word captions synchronized with video audio
"""
import os
from typing import Optional, Callable
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import (
    VideoFileClip, TextClip, CompositeVideoClip,
    ImageClip, concatenate_videoclips
)

from config import (
    CAPTION_FONT_SIZE, CAPTION_FONT_COLOR, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_BG_COLOR, CAPTION_POSITION, CAPTION_MARGIN, TEMP_DIR
)


class CaptionStyle:
    """Configuration for caption styling"""

    def __init__(
        self,
        font_size: int = CAPTION_FONT_SIZE,
        font_color: str = CAPTION_FONT_COLOR,
        highlight_color: str = CAPTION_HIGHLIGHT_COLOR,
        bg_color: tuple = CAPTION_BG_COLOR,
        position: str = CAPTION_POSITION,
        margin: int = CAPTION_MARGIN,
        font_path: Optional[str] = None
    ):
        self.font_size = font_size
        self.font_color = font_color
        self.highlight_color = highlight_color
        self.bg_color = bg_color
        self.position = position
        self.margin = margin
        self.font_path = font_path or self._get_default_font()

    def _get_default_font(self) -> str:
        """Get a default font path that exists on the system"""
        # Common font paths for Windows
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            # Fallbacks
            "arial.ttf",
            "Arial"
        ]
        for font in font_candidates:
            if os.path.exists(font):
                return font
        return "Arial"  # PIL will try to find it


class WordCaptionGenerator:
    """
    Generates word-by-word animated captions for videos.
    Each word is highlighted as it's spoken in the video.
    """

    def __init__(self, style: Optional[CaptionStyle] = None):
        self.style = style or CaptionStyle()
        self.words_per_line = 4  # Maximum words to show at once

    def add_captions_to_video(
        self,
        video_path: str,
        words: list[dict],
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Add word-by-word synchronized captions to a video.

        Args:
            video_path: Path to input video
            words: List of word dicts with 'word', 'start', 'end' keys
            output_path: Output path (auto-generated if None)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the captioned video
        """
        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(
                os.path.dirname(video_path),
                f"{base}_captioned.mp4"
            )

        if progress_callback:
            progress_callback("Loading video for caption overlay...")

        video = VideoFileClip(video_path)

        if progress_callback:
            progress_callback("Generating caption frames...")

        # Group words into display chunks
        word_groups = self._group_words(words)

        # Create caption clips
        caption_clips = []
        for group in word_groups:
            clips = self._create_caption_clips_for_group(
                group, video.size, video.fps
            )
            caption_clips.extend(clips)

        if caption_clips:
            if progress_callback:
                progress_callback("Compositing captions onto video...")

            # Composite all clips
            final_video = CompositeVideoClip([video] + caption_clips)

            if progress_callback:
                progress_callback("Encoding final video with captions...")

            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )

            final_video.close()
        else:
            # No captions to add, just copy
            video.write_videofile(output_path, codec='libx264', audio_codec='aac',
                                 verbose=False, logger=None)

        video.close()

        if progress_callback:
            progress_callback(f"Captioned video saved: {output_path}")

        return output_path

    def _group_words(self, words: list[dict]) -> list[list[dict]]:
        """
        Group words into display chunks based on timing and word count.
        Each group will be displayed together with word highlighting.
        """
        if not words:
            return []

        groups = []
        current_group = []

        for word in words:
            current_group.append(word)

            # Start new group after N words or if there's a pause
            if len(current_group) >= self.words_per_line:
                groups.append(current_group)
                current_group = []
            elif current_group and len(current_group) > 1:
                # Check for pause (gap > 0.5 seconds suggests natural break)
                prev_word = current_group[-2]
                if word['start'] - prev_word['end'] > 0.5:
                    groups.append(current_group[:-1])
                    current_group = [word]

        if current_group:
            groups.append(current_group)

        return groups

    def _create_caption_clips_for_group(
        self,
        word_group: list[dict],
        video_size: tuple,
        fps: float
    ) -> list:
        """Create caption clips for a group of words with highlighting"""
        if not word_group:
            return []

        clips = []
        video_w, video_h = video_size

        # Get timing for the whole group
        group_start = word_group[0]['start']
        group_end = word_group[-1]['end']

        # Create a clip for each word's highlight moment
        for i, current_word in enumerate(word_group):
            # Build the text with current word highlighted
            words_text = [w['word'] for w in word_group]

            # Create image with highlighted word
            img = self._create_caption_image(
                words_text,
                highlight_index=i,
                video_width=video_w
            )

            # Convert to moviepy clip
            img_clip = ImageClip(np.array(img))

            # Position the caption
            y_pos = self._calculate_y_position(video_h, img.height)
            x_pos = (video_w - img.width) // 2

            img_clip = img_clip.set_position((x_pos, y_pos))

            # Set timing - this word's highlight duration
            word_start = current_word['start']
            if i < len(word_group) - 1:
                word_end = word_group[i + 1]['start']
            else:
                word_end = current_word['end']

            img_clip = img_clip.set_start(word_start).set_duration(word_end - word_start)

            clips.append(img_clip)

        return clips

    def _create_caption_image(
        self,
        words: list[str],
        highlight_index: int,
        video_width: int
    ) -> Image.Image:
        """Create a caption image with one word highlighted"""
        # Calculate text dimensions
        try:
            font = ImageFont.truetype(self.style.font_path, self.style.font_size)
        except:
            font = ImageFont.load_default()

        # Build full text
        full_text = " ".join(words)

        # Create a temporary image to measure text
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        # Get text bounding box
        bbox = temp_draw.textbbox((0, 0), full_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Add padding for background
        padding = 20
        img_width = min(text_width + padding * 2, video_width - 40)
        img_height = text_height + padding * 2

        # Create the actual image
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw background with rounded corners
        bg_color = self.style.bg_color
        self._draw_rounded_rectangle(
            draw,
            (0, 0, img_width, img_height),
            radius=15,
            fill=bg_color
        )

        # Draw each word, highlighting the current one
        x_offset = padding
        y_offset = padding

        for i, word in enumerate(words):
            # Determine color
            if i == highlight_index:
                color = self.style.highlight_color
            else:
                color = self.style.font_color

            # Draw the word
            draw.text((x_offset, y_offset), word, font=font, fill=color)

            # Move x position for next word
            word_bbox = draw.textbbox((0, 0), word + " ", font=font)
            x_offset += word_bbox[2] - word_bbox[0]

        return img

    def _draw_rounded_rectangle(
        self,
        draw: ImageDraw.Draw,
        coords: tuple,
        radius: int,
        fill: tuple
    ):
        """Draw a rounded rectangle"""
        x1, y1, x2, y2 = coords

        # Draw main rectangles
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)

        # Draw corners
        draw.ellipse([x1, y1, x1 + radius * 2, y1 + radius * 2], fill=fill)
        draw.ellipse([x2 - radius * 2, y1, x2, y1 + radius * 2], fill=fill)
        draw.ellipse([x1, y2 - radius * 2, x1 + radius * 2, y2], fill=fill)
        draw.ellipse([x2 - radius * 2, y2 - radius * 2, x2, y2], fill=fill)

    def _calculate_y_position(self, video_height: int, caption_height: int) -> int:
        """Calculate Y position based on configured position"""
        if self.style.position == "top":
            return self.style.margin
        elif self.style.position == "center":
            return (video_height - caption_height) // 2
        else:  # bottom
            return video_height - caption_height - self.style.margin


class SimpleCaptionGenerator:
    """
    Simpler caption generator using MoviePy's TextClip.
    Fallback if PIL-based generation has issues.
    """

    def __init__(self, style: Optional[CaptionStyle] = None):
        self.style = style or CaptionStyle()

    def add_captions_to_video(
        self,
        video_path: str,
        words: list[dict],
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Add simple word-by-word captions using TextClip"""
        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(
                os.path.dirname(video_path),
                f"{base}_captioned.mp4"
            )

        if progress_callback:
            progress_callback("Loading video...")

        video = VideoFileClip(video_path)
        video_w, video_h = video.size

        if progress_callback:
            progress_callback("Creating caption clips...")

        caption_clips = []

        for word_data in words:
            try:
                # Create text clip for this word
                txt_clip = TextClip(
                    word_data['word'],
                    fontsize=self.style.font_size,
                    color=self.style.highlight_color,
                    font='Arial',
                    stroke_color='black',
                    stroke_width=2
                )

                # Position at bottom center
                txt_clip = txt_clip.set_position(('center', video_h - self.style.margin))

                # Set timing
                txt_clip = txt_clip.set_start(word_data['start'])
                txt_clip = txt_clip.set_duration(word_data['end'] - word_data['start'])

                caption_clips.append(txt_clip)
            except Exception as e:
                print(f"Warning: Could not create caption for '{word_data['word']}': {e}")
                continue

        if caption_clips:
            if progress_callback:
                progress_callback("Compositing video...")

            final = CompositeVideoClip([video] + caption_clips)

            if progress_callback:
                progress_callback("Encoding output...")

            final.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            final.close()
        else:
            video.write_videofile(output_path, codec='libx264', audio_codec='aac',
                                 verbose=False, logger=None)

        video.close()

        if progress_callback:
            progress_callback(f"Done: {output_path}")

        return output_path


def test_caption_generator():
    """Test the caption generator module"""
    print("Caption generator module loaded successfully")

    style = CaptionStyle()
    print(f"Font path: {style.font_path}")
    print(f"Font size: {style.font_size}")
    print(f"Position: {style.position}")


if __name__ == "__main__":
    test_caption_generator()
