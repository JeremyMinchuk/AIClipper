"""
AI Video Editor - Gradio Web Interface
Upload long-form videos and generate short-form clips with AI-powered captions
"""
import os
import gradio as gr
from pathlib import Path
import shutil

from config import (
    OUTPUT_DIR, TEMP_DIR, SUPPORTED_FORMATS,
    SHORT_MIN_DURATION, SHORT_MAX_DURATION, OLLAMA_MODEL,
    WHISPER_MODEL, MAX_CLIPS_TO_GENERATE
)
from ollama_client import OllamaClipDetector, test_ollama_connection
from video_processor import VideoTranscriber, VideoClipper, ClipProcessor
from caption_generator import WordCaptionGenerator, CaptionStyle


class AIVideoEditor:
    """Main application class that orchestrates all components"""

    def __init__(self):
        self.ollama = OllamaClipDetector()
        self.transcriber = VideoTranscriber()
        self.clipper = VideoClipper()
        self.processor = ClipProcessor()
        self.caption_gen = WordCaptionGenerator()

        # State
        self.current_video_path = None
        self.current_transcript = None
        self.suggested_clips = []

    def check_system_status(self) -> str:
        """Check if all required components are available"""
        status_lines = []

        # Check Ollama
        if self.ollama.check_connection():
            status_lines.append(f"Ollama: Connected ({OLLAMA_MODEL})")
        else:
            status_lines.append(f"Ollama: Not connected")

        # Check directories
        status_lines.append(f"Output: {OUTPUT_DIR}")
        status_lines.append(f"Temp: {TEMP_DIR}")

        # Check Whisper
        status_lines.append(f"Whisper: {WHISPER_MODEL}")

        return "\n".join(status_lines)

    def process_video(
        self,
        video_file,
        num_clips: int,
        add_captions: bool,
        context: str,
        progress=gr.Progress()
    ):
        """
        Main processing pipeline:
        1. Load and transcribe video
        2. Analyze with AI for clip suggestions
        3. Extract clips
        4. Add captions if requested
        """
        if video_file is None:
            return None, "Please upload a video file first.", None

        progress(0, desc="Starting...")

        try:
            # Get video path - handle both string and file object
            if isinstance(video_file, str):
                video_path = video_file
            else:
                video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)

            self.current_video_path = video_path
            print(f"Processing video: {video_path}")

            # Step 1: Transcribe
            progress(0.1, desc="Transcribing audio...")
            self.current_transcript = self.transcriber.transcribe(
                video_path,
                progress_callback=lambda msg: print(msg)
            )

            if not self.current_transcript['segments']:
                return None, "Could not transcribe audio from video.", None

            progress(0.4, desc="Analyzing content with AI...")

            # Step 2: AI Analysis
            context_str = context if context and context.strip() else None
            self.suggested_clips = self.ollama.analyze_transcript_for_clips(
                self.current_transcript['segments'],
                num_clips=int(num_clips),
                context=context_str
            )

            if not self.suggested_clips:
                return None, "AI could not identify engaging clips. Try adjusting the context or check Ollama connection.", None

            progress(0.5, desc=f"Found {len(self.suggested_clips)} potential clips")

            # Step 3: Extract clips
            output_files = []
            total_clips = len(self.suggested_clips)

            for i, clip_data in enumerate(self.suggested_clips):
                progress(
                    0.5 + (0.4 * i / total_clips),
                    desc=f"Extracting clip {i + 1}/{total_clips}..."
                )

                # Extract the clip
                clip_path = self.clipper.extract_clip(
                    video_path=video_path,
                    start_time=clip_data['start'],
                    end_time=clip_data['end'],
                    convert_vertical=True
                )

                # Add captions if requested
                if add_captions and self.current_transcript.get('words'):
                    progress(
                        0.5 + (0.4 * (i + 0.5) / total_clips),
                        desc=f"Adding captions to clip {i + 1}..."
                    )

                    # Get words for this clip's time range
                    clip_words = self.processor.get_words_for_timerange(
                        self.current_transcript['words'],
                        clip_data['start'],
                        clip_data['end']
                    )

                    if clip_words:
                        captioned_path = clip_path.replace('.mp4', '_captioned.mp4')
                        try:
                            clip_path = self.caption_gen.add_captions_to_video(
                                clip_path,
                                clip_words,
                                captioned_path
                            )
                        except Exception as e:
                            print(f"Warning: Could not add captions: {e}")

                output_files.append(clip_path)

            progress(1.0, desc="Complete!")

            # Format results
            results_text = self._format_results(self.suggested_clips, output_files)

            # Return first clip for preview, results text, and file list
            first_clip = output_files[0] if output_files else None
            return first_clip, results_text, output_files

        except Exception as e:
            import traceback
            error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, error_msg, None

    def _format_results(self, clips: list, output_files: list) -> str:
        """Format the results for display"""
        lines = [f"## Generated {len(output_files)} Clips\n"]

        for i, (clip, path) in enumerate(zip(clips, output_files)):
            lines.append(f"### Clip {i + 1}")
            lines.append(f"- **Time:** {clip['start']:.1f}s - {clip['end']:.1f}s")
            lines.append(f"- **Why:** {clip.get('reason', 'N/A')}")
            lines.append(f"- **Hook:** {clip.get('hook', 'N/A')}")
            lines.append(f"- **Score:** {clip.get('score', 'N/A')}/10")
            lines.append(f"- **File:** `{os.path.basename(path)}`")
            lines.append("")

        return "\n".join(lines)

    def get_transcript_text(self) -> str:
        """Return formatted transcript"""
        if not self.current_transcript:
            return "No transcript available. Upload and process a video first."

        lines = ["## Full Transcript\n"]
        for seg in self.current_transcript['segments']:
            time_str = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
            lines.append(f"{time_str} {seg['text']}")

        return "\n".join(lines)


def create_ui():
    """Create the Gradio interface"""
    editor = AIVideoEditor()

    with gr.Blocks(
        title="AI Video Editor",
        theme=gr.themes.Soft()
    ) as app:

        gr.Markdown(
            """
            # AI Video Editor
            ### Transform long-form videos into engaging short-form clips with AI-powered captions

            Upload a video, and the AI will:
            1. **Transcribe** the audio using Whisper
            2. **Analyze** content using your local Ollama model to find engaging moments
            3. **Extract** clips and convert to vertical (9:16) format
            4. **Add captions** with word-by-word highlighting synchronized to speech
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Input Section
                gr.Markdown("### Upload Video")
                video_input = gr.Video(
                    label="Select Video File",
                    sources=["upload"]
                )

                with gr.Row():
                    num_clips = gr.Slider(
                        minimum=1,
                        maximum=MAX_CLIPS_TO_GENERATE,
                        value=3,
                        step=1,
                        label="Number of Clips"
                    )
                    add_captions = gr.Checkbox(
                        value=True,
                        label="Add Word-by-Word Captions"
                    )

                context_input = gr.Textbox(
                    label="Video Context (optional)",
                    placeholder="E.g., 'This is a podcast about technology startups'",
                    lines=2
                )

                process_btn = gr.Button(
                    "Generate Clips",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # Status Section
                gr.Markdown("### System Status")
                status_display = gr.Textbox(
                    value=editor.check_system_status(),
                    label="Status",
                    lines=5,
                    interactive=False
                )

                refresh_btn = gr.Button("Refresh Status")

        gr.Markdown("---")

        # Output Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Preview")
                video_output = gr.Video(
                    label="Generated Clip Preview"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                results_output = gr.Markdown(
                    value="Upload a video and click 'Generate Clips' to start."
                )

        # All generated clips
        gr.Markdown("### All Generated Clips")
        clips_output = gr.Files(
            label="Download Clips"
        )

        # Transcript viewer
        with gr.Accordion("View Full Transcript", open=False):
            transcript_display = gr.Markdown(
                value="Transcript will appear here after processing."
            )
            show_transcript_btn = gr.Button("Show Transcript")

        # Event handlers
        process_btn.click(
            fn=editor.process_video,
            inputs=[video_input, num_clips, add_captions, context_input],
            outputs=[video_output, results_output, clips_output]
        )

        refresh_btn.click(
            fn=editor.check_system_status,
            outputs=[status_display]
        )

        show_transcript_btn.click(
            fn=editor.get_transcript_text,
            outputs=[transcript_display]
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Tips:**
            - For best results, use videos with clear audio
            - The AI works better with context about your video content
            - Processing time depends on video length and your hardware
            - Generated clips are saved in the `output` folder

            *Powered by Whisper (transcription), Ollama (AI analysis), and MoviePy (video processing)*
            """
        )

    return app


def main():
    """Launch the application"""
    print("=" * 50)
    print("AI Video Editor")
    print("=" * 50)

    # Check Ollama connection
    print("\nChecking Ollama connection...")
    if test_ollama_connection():
        print("Ollama is ready")
    else:
        print("WARNING: Ollama not detected - some features may not work")
        print(f"  Run: ollama serve")
        print(f"  Then: ollama pull {OLLAMA_MODEL}")

    # Create and launch UI
    print("\nStarting web interface...")
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    main()
