from datasets import load_dataset

import random
import numpy as np

import torch
import torch.nn as nn

from encoder import encode_text


NUM_USERS = 5000
COMMENTS_PER_USER = 20
EPOCHS = 10


print("Loading dataset...")

ds = load_dataset(
    "jingjietan/pandora-big5"
)

train = ds["train"]


print("Grouping comments by user...")

users = {}

for row in train:

    ptype = int(row["ptype"])

    if ptype not in users:

        users[ptype] = {
            "texts": [],
            "traits": [
                row["O"] / 100.0,
                row["C"] / 100.0,
                row["E"] / 100.0,
                row["A"] / 100.0,
                row["N"] / 100.0,
            ]
        }

    users[ptype]["texts"].append(
        row["text"]
    )


print(
    f"Found {len(users)} users"
)

user_ids = list(
    users.keys()
)

random.shuffle(
    user_ids
)

X = []
Y = []

print(
    "Building user profiles..."
)

for i, user_id in enumerate(
    user_ids[:NUM_USERS]
):

    texts = users[user_id]["texts"]

    if len(texts) < COMMENTS_PER_USER:
        continue

    sampled = random.sample(
        texts,
        COMMENTS_PER_USER
    )

    combined_text = " ".join(
        sampled
    )

    embedding = encode_text(
        combined_text
    )

    X.append(
        embedding
    )

    Y.append(
        users[user_id]["traits"]
    )

    if i % 100 == 0:
        print(i)


X = np.array(
    X,
    dtype=np.float32
)

Y = np.array(
    Y,
    dtype=np.float32
)

X = torch.tensor(X)
Y = torch.tensor(Y)


class UserTraitPredictor(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                384,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                5
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        x
    ):
        return self.network(x)


model = UserTraitPredictor()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()


print(
    "Training..."
)

for epoch in range(
    EPOCHS
):

    optimizer.zero_grad()

    predictions = model(
        X
    )

    loss = loss_fn(
        predictions,
        Y
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch + 1}: "
        f"{loss.item():.6f}"
    )


torch.save(
    model.state_dict(),
    "user_trait_model.pth"
)

print(
    "Saved model."
)