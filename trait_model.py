"""Trait prediction model: text -> OCEAN (Big Five) trait scores.

Contains the model architecture, lazy model loading, and a ``predict`` helper
that maps a piece of text to a dictionary of five trait scores in [0, 1].
"""

import os

import torch
import torch.nn as nn

from encoder import encode_text, EMBEDDING_DIM

MODEL_PATH = "trait_model.pth"

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TraitPredictor(nn.Module):
    """MLP mapping a MiniLM sentence embedding to five OCEAN scores."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(TRAITS)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


_model = None


def load_model(model_path=MODEL_PATH):
    """Load and cache the trained model in evaluation mode."""
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Train it first with: python train_trait_model.py"
        )

    model = TraitPredictor().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    _model = model
    return _model


def predict(text):
    """Predict OCEAN trait scores for a piece of text.

    Returns a dict mapping each trait name to a score in [0, 1], e.g.::

        predict("I love meeting new people.")
        # {"openness": 0.61, "conscientiousness": 0.48, ...}
    """
    model = load_model()

    embedding = encode_text(text)

    x = torch.tensor(
        embedding,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        scores = model(x).squeeze(0).tolist()

    return dict(zip(TRAITS, scores))
