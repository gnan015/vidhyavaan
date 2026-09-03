from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("Embedding model loaded.")

    def encode(self, texts):

        return self.model.encode(
            texts,
            show_progress_bar=True
        )