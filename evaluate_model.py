import torch
import torch.nn as nn
import numpy as np

from datasets import load_dataset
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from encoder import encode_text


class TraitPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


print("Loading model...")

model = TraitPredictor()

model.load_state_dict(
    torch.load(
        "trait_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Loading dataset...")

dataset = load_dataset(
    "jingjietan/pandora-big5",
    split="validation"
)

dataset = dataset.select(
    range(5000)
)

predictions = []
targets = []

print("Evaluating...")

for i, row in enumerate(dataset):

    if i % 100 == 0:
        print(i)

    embedding = encode_text(
        row["text"]
    )

    x = torch.tensor(
        embedding,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        pred = model(
            x
        )[0].numpy()

    target = np.array([
        row["O"] / 100,
        row["C"] / 100,
        row["E"] / 100,
        row["A"] / 100,
        row["N"] / 100
    ])

    predictions.append(pred)
    targets.append(target)

predictions = np.array(
    predictions
)

targets = np.array(
    targets
)

traits = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism"
]

print("\nRESULTS\n")

for i, trait in enumerate(traits):

    mae = mean_absolute_error(
        targets[:, i],
        predictions[:, i]
    )

    r2 = r2_score(
        targets[:, i],
        predictions[:, i]
    )

    print(f"{trait}")
    print(f"MAE: {mae:.4f}")
    print(f"R² : {r2:.4f}")
    print("-" * 30)