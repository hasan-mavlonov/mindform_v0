"""Train the affect/appraisal head from data/appraisal_dataset.jsonl.

MiniLM embeddings -> small regression head, with the appraisal vector as the
target (the causal mediator of formation), NOT author trait scores. Run locally
once the dataset exists (needs torch + sentence-transformers):

    python bootstrap/build_affect_dataset.py
    python bootstrap/train_appraisal_head.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from encoder import encode_text
from appraisal_head import AppraisalHead, OUT_KEYS, HEAD_PATH, squash

DATA = "data/appraisal_dataset.jsonl"


def target_vector(appraisal):
    return [appraisal.get(k, 0.0) for k in OUT_KEYS]


def main(data=DATA, epochs=300, lr=1e-3):
    rows = [json.loads(line) for line in open(data) if line.strip()]
    print(f"Encoding {len(rows)} samples with MiniLM...")
    X = torch.tensor(encode_text([r["text"] for r in rows]), dtype=torch.float32)
    Y = torch.tensor([target_vector(r["appraisal"]) for r in rows], dtype=torch.float32)

    model = AppraisalHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(squash(model(X)), Y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")

    torch.save(model.state_dict(), HEAD_PATH)
    print(f"saved -> {HEAD_PATH}")


if __name__ == "__main__":
    main()
