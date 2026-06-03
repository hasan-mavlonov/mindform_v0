"""
Text encoder backed by MiniLM sentence-transformer.

Produces 384-dimensional embeddings for trait prediction model.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Model config
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Load model ONCE
model = SentenceTransformer(MODEL_NAME)


def encode_text(texts, batch_size=32):
    """
    Encode a list of texts into embeddings.

    Args:
        texts (list[str]): input sentences
        batch_size (int): encoding batch size

    Returns:
        np.ndarray: shape (N, 384)
    """

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    total = len(texts)

    print(f"Encoding {total} texts...", flush=True)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]

        batch_emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings.append(batch_emb)

        print(
            f"Encoded {min(i + batch_size, total)}/{total}",
            flush=True
        )

    print("Encoding complete.", flush=True)

    return np.vstack(embeddings)