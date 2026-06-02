import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset
from encoder import encode_text

MODEL_PATH = "trait_model.pth"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SAMPLE_SIZE = 50_000
BATCH_SIZE = 256
EPOCHS = 10


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


print("Loading dataset...")

ds = load_dataset(
    "jingjietan/pandora-big5"
)

train = ds["train"].select(
    range(SAMPLE_SIZE)
)

print("Encoding texts...")

embeddings = []
targets = []

for i, sample in enumerate(train):

    embedding = encode_text(
        sample["text"]
    )

    target = [
        sample["O"] / 100.0,
        sample["C"] / 100.0,
        sample["E"] / 100.0,
        sample["A"] / 100.0,
        sample["N"] / 100.0,
    ]

    embeddings.append(embedding)
    targets.append(target)

    if i % 1000 == 0:
        print(
            f"{i}/{SAMPLE_SIZE}"
        )

X = torch.tensor(
    embeddings,
    dtype=torch.float32
)

y = torch.tensor(
    targets,
    dtype=torch.float32
)

dataset = TensorDataset(
    X,
    y
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = TraitPredictor().to(
    DEVICE
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training...")

for epoch in range(EPOCHS):

    total_loss = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )

        predictions = model(
            batch_x
        )

        loss = criterion(
            predictions,
            batch_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(loader)
    )

    print(
        f"Epoch {epoch + 1}: "
        f"{avg_loss:.6f}"
    )

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    f"Saved model to {MODEL_PATH}"
)