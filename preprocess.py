"""Audio preprocessing for Whisper fine-tuning.

Handles resampling, noise augmentation, and silence trimming
using Librosa for robust ASR training.

Usage:
    python preprocess.py --audio path/to/audio.wav
"""

import argparse
import numpy as np
import librosa
import soundfile as sf


def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sample rate.

    Args:
        audio: Audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate (16kHz for Whisper).

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def add_noise(audio, snr_db=20):
    """Add Gaussian noise at specified SNR level.

    Args:
        audio: Input audio array.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Noisy audio array.
    """
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


def trim_silence(audio, sr=16000, top_db=25):
    """Trim leading and trailing silence from audio.

    Args:
        audio: Input audio array.
        sr: Sample rate.
        top_db: Threshold in dB below peak to consider as silence.

    Returns:
        Trimmed audio array.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize_audio(audio):
    """Peak normalize audio to [-1, 1] range.

    Args:
        audio: Input audio array.

    Returns:
        Normalized audio array.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def preprocess_audio(audio, sr, target_sr=16000, augment=False, trim=True, snr_range=(10, 30)):
    """Full preprocessing pipeline for Whisper input.

    Args:
        audio: Input audio array.
        sr: Original sample rate.
        target_sr: Target sample rate.
        augment: Whether to apply noise augmentation.
        trim: Whether to trim silence.
        snr_range: SNR range for noise augmentation.

    Returns:
        Preprocessed audio array at target sample rate.
    """
    # Resample to 16kHz
    audio = resample_audio(audio, sr, target_sr)

    # Trim silence
    if trim:
        audio = trim_silence(audio, target_sr)

    # Noise augmentation (training only)
    if augment:
        snr_db = np.random.uniform(*snr_range)
        audio = add_noise(audio, snr_db)

    # Normalize
    audio = normalize_audio(audio)

    return audio


def get_audio_features(audio, sr=16000):
    """Extract audio features for analysis.

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        Dict with audio statistics.
    """
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    # Mel spectrogram stats
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return {
        "duration_s": duration,
        "sample_rate": sr,
        "samples": len(audio),
        "rms_energy": float(rms),
        "zero_crossing_rate": float(zcr),
        "mel_mean_db": float(mel_db.mean()),
        "mel_std_db": float(mel_db.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for Whisper")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-trim", action="store_true")
    args = parser.parse_args()

    audio, sr = librosa.load(args.audio, sr=None)
    print(f"Input: {args.audio}")
    print(f"  Duration: {len(audio)/sr:.2f}s, SR: {sr}Hz")

    features_before = get_audio_features(audio, sr)

    processed = preprocess_audio(
        audio, sr,
        augment=args.augment,
        trim=not args.no_trim,
    )

    features_after = get_audio_features(processed, 16000)

    print(f"\nAfter preprocessing:")
    print(f"  Duration: {features_after['duration_s']:.2f}s, SR: 16000Hz")
    print(f"  RMS Energy: {features_after['rms_energy']:.4f}")

    if args.output:
        sf.write(args.output, processed, 16000)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
