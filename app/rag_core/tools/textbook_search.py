from rag.retriever import Retriever


class TextbookSearchTool:

    def __init__(self):

        self.retriever = Retriever()

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        print("\nSearching textbook...")

        results = self.retriever.search(
            query,
            top_k=top_k
        )

        return results