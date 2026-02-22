"""Transcribe Vietnamese audio using fine-tuned Whisper.

Usage:
    python transcribe.py --audio path/to/audio.wav
"""

import argparse
import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from preprocess import preprocess_audio


def load_model(model_dir="outputs/model"):
    """Load fine-tuned Whisper model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    try:
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    except Exception:
        print("Fine-tuned model not found. Using base Whisper-small...")
        model_name = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    model.eval()
    return model, processor, device


def transcribe(audio_path, model, processor, device, language="vi"):
    """Transcribe a single audio file.

    Args:
        audio_path: Path to audio file.
        model: Whisper model.
        processor: Whisper processor.
        device: torch device.
        language: Language code for decoding.

    Returns:
        Transcription string.
    """
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=None)
    audio = preprocess_audio(audio, sr, augment=False, trim=True)

    # Process with Whisper
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    # Generate transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=128,
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def transcribe_array(audio_array, sr, model, processor, device, language="vi"):
    """Transcribe from audio array (for Gradio/mic input).

    Args:
        audio_array: Numpy audio array.
        sr: Sample rate.
        model: Whisper model.
        processor: Whisper processor.
        device: torch device.
        language: Language code.

    Returns:
        Transcription string.
    """
    audio = preprocess_audio(audio_array.astype(np.float32), sr, augment=False, trim=True)

    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=128,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe Vietnamese audio")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="outputs/model")
    parser.add_argument("--language", type=str, default="vi")
    args = parser.parse_args()

    model, processor, device = load_model(args.model_dir)

    audio, sr = librosa.load(args.audio, sr=None)
    print(f"Audio: {args.audio} ({len(audio)/sr:.2f}s, {sr}Hz)")

    transcription = transcribe(args.audio, model, processor, device, args.language)
    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    main()
