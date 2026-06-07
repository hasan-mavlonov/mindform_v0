"""Learned affect/appraisal head: frozen MiniLM embedding -> appraisal vector.

Outputs the ``config.APPRAISAL_DIMS``. Unit dims pass through a sigmoid, signed
dims through a tanh. Trained on existing affect/appraisal corpora (VAD / emotion /
appraisal) by ``bootstrap/train_appraisal_head.py`` -- a frozen MiniLM-embedding +
small regression-head fit on the appraisal labels.
"""

import os

import torch
import torch.nn as nn

from encoder import encode_text, EMBEDDING_DIM
from config import APPRAISAL_DIMS, APPRAISAL_SCHEMA

HEAD_PATH = "appraisal_head.pth"
OUT_KEYS = list(APPRAISAL_DIMS)

# 1.0 where the output is a unit [0, 1] dim, 0.0 where it is signed [-1, 1].
_UNIT = torch.tensor(
    [1.0 if APPRAISAL_SCHEMA.get(k) == "unit" else 0.0 for k in OUT_KEYS]
)


class AppraisalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(OUT_KEYS)),
        )

    def forward(self, x):
        return self.net(x)


def squash(raw):
    """Map raw logits to per-dim ranges. Batched (used in training too)."""
    return _UNIT * torch.sigmoid(raw) + (1 - _UNIT) * torch.tanh(raw)


_head = None


def load_head(path=HEAD_PATH):
    """Load and cache the head, or return None if it has not been trained."""
    global _head
    if _head is not None:
        return _head
    if not os.path.exists(path):
        return None

    model = AppraisalHead()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    _head = model
    return _head


def predict_appraisal(text):
    """Return the appraisal dict from the learned head."""
    model = load_head()
    x = torch.tensor(encode_text(text), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = squash(model(x))[0].tolist()
    return dict(zip(OUT_KEYS, out))
