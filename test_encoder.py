from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    """
    I love being around people.
    I enjoy parties.
    Meeting strangers excites me.
    """,

    """
    I spend most weekends reading.
    Large crowds exhaust me.
    I prefer quiet environments.
    """,

    """
    I like solving difficult technical problems.
    I enjoy learning mathematics.
    Complex systems fascinate me.
    """
]

embeddings = encoder.encode(texts)

print("Social vs Introvert:",
      cos_sim(embeddings[0], embeddings[1]))

print("Social vs Intellectual:",
      cos_sim(embeddings[0], embeddings[2]))

print("Introvert vs Intellectual:",
      cos_sim(embeddings[1], embeddings[2]))