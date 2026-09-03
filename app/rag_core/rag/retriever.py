from models.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


class Retriever:

    def __init__(self):

        print("Initializing retriever...")

        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

        print("Retriever ready.")

    def search(self, question: str, top_k: int = 5):

        question_embedding = self.embedding_model.encode(
            [question]
        )

        results = self.vector_store.collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=top_k
        )

        return results

if __name__ == "__main__":

    retriever = Retriever()

    question = input("\nAsk a question: ")

    results = retriever.search(question)

    print("\n================================")
    print("RETRIEVED RESULTS")
    print("================================")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        print(f"\nResult {i + 1}")
        print("-----------------------------")

        print("Distance:", distances[i])

        print("Book:", metadatas[i]["book"])

        print("Page:", metadatas[i]["page"])

        print("\nText:")
        print(documents[i])