"""Text encoder backed by a MiniLM sentence-transformer.

Produces 384-dimensional embeddings that serve as the input features for the
trait prediction model. Accepts either a single string or a list of strings
(lists are batched automatically by sentence-transformers).
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_encoder = SentenceTransformer(MODEL_NAME)


def encode_text(text):
    """Return the MiniLM embedding(s) for a string or list of strings.

    A single string yields a 1-D array of shape (384,); a list of strings
    yields a 2-D array of shape (len(text), 384).
    """
    return _encoder.encode(text)
