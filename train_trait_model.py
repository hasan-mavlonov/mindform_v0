"""Production training script for the MindForm trait prediction model.

Trains a model that maps text -> OCEAN (Big Five) trait scores using MiniLM
sentence embeddings. The data is split into train/validation sets, both losses
are reported each epoch, and the model is only saved when validation loss
improves.

Usage:
    python train_trait_model.py
    python train_trait_model.py --sample-size 100000 --epochs 20
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from datasets import load_dataset

from encoder import encode_text
from trait_model import TraitPredictor, MODEL_PATH, DEVICE

DATASET_NAME = "jingjietan/pandora-big5"

# Pandora label columns in OCEAN order, scaled from 0-100 to 0-1.
LABEL_COLUMNS = ["O", "C", "E", "A", "N"]


def build_dataset(sample_size):
    """Encode Pandora samples into a TensorDataset of (embedding, target)."""
    print(f"Loading dataset '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME, split="train")

    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))

    print(f"Encoding {len(ds)} samples with MiniLM...")
    embeddings = encode_text(ds["text"])

    # Fetch each label column once, then scale 0-100 -> 0-1 per row.
    columns = [ds[col] for col in LABEL_COLUMNS]
    targets = [[value / 100.0 for value in row] for row in zip(*columns)]

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    return TensorDataset(X, y)


def split_dataset(dataset, val_fraction, seed=42):
    """Split a dataset into (train, validation) subsets."""
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(model, loader, criterion, optimizer=None):
    """Run one epoch. Trains when an optimizer is given, otherwise evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)


def train(sample_size, epochs, batch_size, lr, val_fraction, model_path):
    dataset = build_dataset(sample_size)
    train_set, val_set = split_dataset(dataset, val_fraction)

    print(f"Train samples: {len(train_set)} | Validation samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = TraitPredictor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {DEVICE} for {epochs} epochs...\n")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f}"
            f"{'  <- saved' if improved else ''}"
        )

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MindForm trait prediction model (text -> OCEAN)."
    )
    parser.add_argument("--sample-size", type=int, default=50_000,
                        help="Number of samples to use (0 = full dataset).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
