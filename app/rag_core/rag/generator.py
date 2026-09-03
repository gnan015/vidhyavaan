from models.llm import GroqLLM
from rag.prompts import build_rag_prompt


class RAGGenerator:

    def __init__(self):

        self.llm = GroqLLM()

    def generate_from_rag(
        self,
        question: str,
        retrieved_documents,
        conversation_history=None
    ):

        context_parts = []

        for document, metadata in zip(
            retrieved_documents["documents"][0],
            retrieved_documents["metadatas"][0]
        ):

            context_parts.append(
                f"""
Book: {metadata['book']}
Page: {metadata['page']}

Content:
{document}
"""
            )

        context = "\n".join(context_parts)

        prompt = build_rag_prompt(
            question,
            context
        )

        answer = self.llm.generate(
            prompt,
            conversation_history
        )

        return answer

    def generate_general(
        self,
        question: str,
        conversation_history=None
    ):

        prompt = f"""
You are a helpful academic AI assistant.

The student's academic textbooks do not contain
sufficient information to answer this question.

Answer the question using your general knowledge.

Question:
{question}

Give a clear and educational explanation.
If you are uncertain about something, say so.
"""

        return self.llm.generate(
            prompt,
            conversation_history
        )