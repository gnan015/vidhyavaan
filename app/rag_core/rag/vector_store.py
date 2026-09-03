import chromadb

from config import VECTOR_DB_DIR, COLLECTION_NAME


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def add_documents(self, documents, embeddings):

        ids = [
            f"chunk_{i}"
            for i in range(len(documents))
        ]

        texts = [
            document["text"]
            for document in documents
        ]

        metadatas = [
            document["metadata"]
            for document in documents
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(f"Added {len(documents)} chunks to ChromaDB.")

    def count(self):

        return self.collection.count()