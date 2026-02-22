"""Gradio demo for Vietnamese speech recognition with Whisper.

Supports both microphone input and file upload.

Usage:
    python app.py
"""

import gradio as gr
import numpy as np
from transcribe import load_model, transcribe, transcribe_array


# Load model at startup
model, processor, device = load_model()


def transcribe_audio(audio):
    """Transcribe audio input from Gradio.

    Args:
        audio: Tuple of (sample_rate, audio_array) from Gradio Audio component.

    Returns:
        Transcription text.
    """
    if audio is None:
        return "No audio input provided."

    sr, audio_array = audio

    # Convert to float32 and mono
    if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0

    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    transcription = transcribe_array(audio_array, sr, model, processor, device, language="vi")
    return transcription


def transcribe_file(filepath):
    """Transcribe uploaded audio file.

    Args:
        filepath: Path to uploaded audio file.

    Returns:
        Transcription text.
    """
    if filepath is None:
        return "No file uploaded."

    transcription = transcribe(filepath, model, processor, device, language="vi")
    return transcription


# Gradio interface with tabs for mic and file input
with gr.Blocks(title="Vietnamese Speech Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Vietnamese Speech Recognition with Whisper\n"
        "Fine-tuned **OpenAI Whisper** on Mozilla Common Voice Vietnamese data for ASR. "
        "Supports both **microphone recording** and **file upload**."
    )

    with gr.Tabs():
        with gr.TabItem("Microphone"):
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
            mic_output = gr.Textbox(label="Transcription", lines=3)
            mic_button = gr.Button("Transcribe", variant="primary")
            mic_button.click(transcribe_audio, inputs=mic_input, outputs=mic_output)

        with gr.TabItem("File Upload"):
            file_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
            file_output = gr.Textbox(label="Transcription", lines=3)
            file_button = gr.Button("Transcribe", variant="primary")
            file_button.click(transcribe_file, inputs=file_input, outputs=file_output)

    gr.Markdown(
        "### Details\n"
        "- **Model**: Whisper-small fine-tuned on Vietnamese Common Voice\n"
        "- **Preprocessing**: Resampling to 16kHz, noise augmentation, silence trimming\n"
        "- **Metrics**: Evaluated with Word Error Rate (WER) and Character Error Rate (CER)\n"
    )


if __name__ == "__main__":
    demo.launch(share=False)
