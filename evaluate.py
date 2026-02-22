"""Evaluate Vietnamese ASR with WER and CER metrics.

Usage:
    python evaluate.py
"""

import argparse
import os
import numpy as np
import yaml
from jiwer import wer, cer
from tqdm import tqdm

from transcribe import load_model, transcribe_array


def evaluate_asr(model, processor, device, test_data, language="vi"):
    """Evaluate ASR model on test data.

    Args:
        model: Whisper model.
        processor: Whisper processor.
        device: torch device.
        test_data: List of dicts with 'audio' and 'sentence'.
        language: Language code.

    Returns:
        Dict with WER, CER, and per-sample results.
    """
    references = []
    hypotheses = []

    for item in tqdm(test_data, desc="Evaluating"):
        if isinstance(item["audio"], dict):
            audio = np.array(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"]["sampling_rate"]
        else:
            audio = np.array(item["audio"], dtype=np.float32)
            sr = 16000

        transcription = transcribe_array(audio, sr, model, processor, device, language)
        reference = item["sentence"].strip()

        references.append(reference)
        hypotheses.append(transcription)

    # Compute metrics
    word_error_rate = wer(references, hypotheses)
    char_error_rate = cer(references, hypotheses)

    return {
        "wer": word_error_rate,
        "cer": char_error_rate,
        "references": references,
        "hypotheses": hypotheses,
        "n_samples": len(references),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese ASR")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-dir", type=str, default="outputs/model")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    model, processor, device = load_model(args.model_dir)

    # Load test data
    try:
        from data.prepare_common_voice import load_common_voice
        _, test_data = load_common_voice("vi", args.max_samples)
        test_samples = [{"audio": item["audio"], "sentence": item["sentence"]} for item in test_data]
    except Exception:
        print("Using synthetic data for evaluation...")
        from data.prepare_common_voice import generate_synthetic_audio
        test_samples = generate_synthetic_audio(n_samples=20)

    results = evaluate_asr(model, processor, device, test_samples[:args.max_samples])

    print("\n" + "=" * 50)
    print("VIETNAMESE ASR EVALUATION")
    print("=" * 50)
    print(f"Samples evaluated: {results['n_samples']}")
    print(f"Word Error Rate (WER): {results['wer']:.4f} ({results['wer']*100:.1f}%)")
    print(f"Character Error Rate (CER): {results['cer']:.4f} ({results['cer']*100:.1f}%)")

    # Show sample results
    print("\nSample Transcriptions:")
    for i in range(min(5, len(results["references"]))):
        print(f"\n  Reference:  {results['references'][i]}")
        print(f"  Hypothesis: {results['hypotheses'][i]}")


if __name__ == "__main__":
    main()
