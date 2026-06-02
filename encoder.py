from sentence_transformers import SentenceTransformer

_encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def encode_text(text):
    return _encoder.encode(text)