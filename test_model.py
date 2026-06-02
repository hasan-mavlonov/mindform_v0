import torch
import torch.nn as nn

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


model = TraitPredictor()

model.load_state_dict(
    torch.load(
        "trait_model.pth",
        map_location="cpu"
    )
)

model.eval()


def predict(text):

    embedding = encode_text(text)

    x = torch.tensor(
        embedding,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        prediction = model(
            x
        )[0]

    traits = {
        "openness":
            float(prediction[0]),

        "conscientiousness":
            float(prediction[1]),

        "extraversion":
            float(prediction[2]),

        "agreeableness":
            float(prediction[3]),

        "neuroticism":
            float(prediction[4]),
    }

    return traits


examples = [

    "I love attending parties and meeting new people.",

    "I spend most evenings reading philosophy books alone.",

    "I feel anxious all the time and worry about everything.",

    "I carefully plan my day and follow schedules.",

    "I enjoy helping others solve their problems."
]
test_cases = [
    "PARTY PARTY PARTY PARTY PARTY PARTY",
    "I am extremely outgoing and love social interaction",
    "I hate being around people",
    "I am terrified of everything",
    "I read philosophy and explore abstract ideas every day",
]

for text in test_cases:

    print("\nTEXT:")
    print(text)

    print("\nPREDICTION:")
    print(
        predict(text)
    )

    print(
        "-" * 60
    )

for text in examples:

    print("\nTEXT:")
    print(text)

    print("\nPREDICTION:")
    print(
        predict(text)
    )

    print(
        "-" * 60
    )