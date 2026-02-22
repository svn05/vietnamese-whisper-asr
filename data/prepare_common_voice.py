"""Prepare Mozilla Common Voice Vietnamese dataset for Whisper fine-tuning.

Downloads and preprocesses the Vietnamese split of Common Voice.

Usage:
    python data/prepare_common_voice.py
    python data/prepare_common_voice.py --max-samples 5000
"""

import argparse
import os
import numpy as np

DATA_DIR = os.path.dirname(__file__)


def load_common_voice(language="vi", max_samples=10000):
    """Load Common Voice dataset from HuggingFace.

    Args:
        language: Language code.
        max_samples: Maximum samples to load.

    Returns:
        Train and test dataset splits.
    """
    from datasets import load_dataset, Audio

    print(f"Loading Common Voice ({language})...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        language,
        split="train",
        trust_remote_code=True,
    )

    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Split into train/test
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")

    return split["train"], split["test"]


def generate_synthetic_audio(n_samples=100, sr=16000, duration=3.0):
    """Generate synthetic audio data for testing pipeline.

    Creates simple sine wave audio with Vietnamese text labels.

    Args:
        n_samples: Number of samples to generate.
        sr: Sample rate.
        duration: Audio duration in seconds.

    Returns:
        List of dicts with 'audio' and 'sentence' keys.
    """
    np.random.seed(42)

    sentences = [
        "xin chào các bạn",
        "hôm nay thời tiết rất đẹp",
        "tôi đang học tiếng việt",
        "cảm ơn bạn rất nhiều",
        "việt nam là một đất nước xinh đẹp",
        "trí tuệ nhân tạo đang phát triển",
        "tôi muốn trở thành kỹ sư",
        "đại học toronto rất tuyệt vời",
        "chúng tôi đang làm dự án mới",
        "xin hãy giúp tôi với bài tập này",
    ]

    samples = []
    for i in range(n_samples):
        # Generate simple audio with varying frequencies
        t = np.linspace(0, duration, int(sr * duration))
        freq = np.random.uniform(200, 800)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * np.random.randn(len(t))
        audio = audio.astype(np.float32)

        sentence = sentences[i % len(sentences)]
        samples.append({
            "audio": {"array": audio, "sampling_rate": sr},
            "sentence": sentence,
        })

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Common Voice Vietnamese data")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic audio data...")
        samples = generate_synthetic_audio(n_samples=100)
        print(f"Generated {len(samples)} synthetic samples")
    else:
        train, test = load_common_voice("vi", args.max_samples)
        print(f"Dataset ready: {len(train)} train, {len(test)} test")
