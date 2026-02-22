"""Fine-tune OpenAI's Whisper on Vietnamese Common Voice data.

Trains Whisper for Vietnamese automatic speech recognition (ASR)
with audio preprocessing via Librosa.

Usage:
    python train.py
    python train.py --epochs 5 --batch-size 8 --lr 1e-5
"""

import argparse
import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from preprocess import preprocess_audio


class WhisperASRDataset(Dataset):
    """Dataset for Whisper fine-tuning on speech data."""

    def __init__(self, audio_data, processor, augment=False):
        """
        Args:
            audio_data: List of dicts with 'audio' and 'sentence' keys.
            processor: WhisperProcessor.
            augment: Whether to apply noise augmentation.
        """
        self.audio_data = audio_data
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        item = self.audio_data[idx]

        # Get audio array
        if isinstance(item["audio"], dict):
            audio = np.array(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"]["sampling_rate"]
        else:
            audio = np.array(item["audio"], dtype=np.float32)
            sr = 16000

        # Preprocess audio
        audio = preprocess_audio(audio, sr, augment=self.augment)

        # Process with Whisper processor
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        # Tokenize transcription
        labels = self.processor.tokenizer(
            item["sentence"],
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        ).input_ids.squeeze(0)

        # Replace padding with -100 for loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "labels": labels,
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_features.size(0)
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Vietnamese ASR")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Whisper model and processor
    model_name = config["model"]["name"]
    print(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    # Set language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )
    model.config.suppress_tokens = []

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Load data
    try:
        from data.prepare_common_voice import load_common_voice
        train_data, test_data = load_common_voice("vi", config["data"]["max_train_samples"])
        train_samples = [{"audio": item["audio"], "sentence": item["sentence"]} for item in train_data]
    except Exception as e:
        print(f"Could not load Common Voice: {e}")
        print("Using synthetic data for demo...")
        from data.prepare_common_voice import generate_synthetic_audio
        train_samples = generate_synthetic_audio(n_samples=100)

    # Create dataset
    train_dataset = WhisperASRDataset(train_samples, processor, augment=True)

    batch_size = args.batch_size or config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    epochs = args.epochs or config["training"]["epochs"]
    lr = args.lr or config["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    output_dir = config["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch}/{epochs} — Loss: {train_loss:.4f}")

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
