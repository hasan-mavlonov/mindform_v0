import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset

from encoder import encode_text
from trait_model import TraitPredictor, MODEL_PATH, DEVICE

DATASET_NAME = "jingjietan/pandora-big5"
LABEL_COLUMNS = ["O", "C", "E", "A", "N"]

import os
import torch

CACHE_PATH = "embeddings_cache.pt"


def build_dataset(sample_size):
    print(f"Loading dataset '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME, split="train")

    if sample_size and sample_size > 0:
        ds = ds.select(range(min(sample_size, len(ds))))

    print(f"Dataset loaded: {len(ds)} samples")

    # ✅ CACHE CHECK
    if os.path.exists(CACHE_PATH):
        print("Loading cached embeddings...")
        embeddings = torch.load(CACHE_PATH)
    else:
        print("Starting encoding (first run, this will take time)...")
        embeddings = encode_text(ds["text"])
        print("Saving embeddings cache...")
        torch.save(embeddings, CACHE_PATH)

    print("Encoding ready.")

    columns = [ds[col] for col in LABEL_COLUMNS]
    targets = [[value / 100.0 for value in row] for row in zip(*columns)]

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    return TensorDataset(X, y)


def split_dataset(dataset, val_fraction, seed=42):
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(model, loader, criterion, optimizer=None, log_every=10, epoch=0):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for step, (batch_x, batch_y) in enumerate(loader, start=1):

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

            if is_train and step % log_every == 0:
                print(
                    f"[Epoch {epoch}] Step {step}/{len(loader)} "
                    f"| batch loss: {loss.item():.6f}"
                )

    return total_loss / len(loader.dataset)


def train(sample_size, epochs, batch_size, lr, val_fraction, model_path):
    dataset = build_dataset(sample_size)
    train_set, val_set = split_dataset(dataset, val_fraction)

    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print(f"Batches per epoch: {len(train_loader)}")

    model = TraitPredictor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    print(f"Training on {DEVICE}\n")

    for epoch in range(1, epochs + 1):

        print(f"\n--- Epoch {epoch} ---")

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            log_every=10,
            epoch=epoch
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            epoch=epoch
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            saved = True
        else:
            saved = False

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train: {train_loss:.6f} | "
            f"val: {val_loss:.6f}"
            f"{'  <- saved' if saved else ''}"
        )

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved model: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample-size", type=int, default=50000)
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
