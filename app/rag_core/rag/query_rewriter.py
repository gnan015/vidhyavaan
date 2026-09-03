from models.llm import GroqLLM


class QueryRewriter:

    def __init__(self):
        self.llm = GroqLLM()

    def rewrite(
        self,
        question: str,
        conversation_history=None
    ):

        return self.llm.rewrite_query(
            question,
            conversation_history
        )