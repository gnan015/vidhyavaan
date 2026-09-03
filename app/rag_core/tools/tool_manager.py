from tools.calculator import Calculator
from tools.textbook_search import TextbookSearchTool

from rag.query_rewriter import QueryRewriter
from rag.evaluator import RAGEvaluator
from rag.generator import RAGGenerator


class ToolManager:

    def __init__(self):

        # -----------------------------------------
        # Tools
        # -----------------------------------------

        self.calculator = Calculator()

        self.textbook_search = TextbookSearchTool()

        # -----------------------------------------
        # RAG components
        # -----------------------------------------

        self.query_rewriter = QueryRewriter()

        self.evaluator = RAGEvaluator(
            threshold=1.2
        )

        self.generator = RAGGenerator()

    # =============================================
    # CALCULATOR
    # =============================================

    def calculate(self, question: str):

        return self.calculator.calculate(
            question
        )

    # =============================================
    # TEXTBOOK SEARCH
    # =============================================

    def search_textbook(
        self,
        query: str,
        top_k: int = 5
    ):

        return self.textbook_search.search(
            query,
            top_k
        )

    # =============================================
    # COMPLETE ACADEMIC RAG
    # =============================================

    def answer_from_textbook(
        self,
        question: str,
        conversation_history=None
    ):

        # -----------------------------------------
        # Rewrite question
        # -----------------------------------------

        rewritten_question = (
            self.query_rewriter.rewrite(
                question,
                conversation_history
            )
        )

        print("\nOriginal question:")
        print(question)

        print("\nSearch query:")
        print(rewritten_question)

        # -----------------------------------------
        # Search textbook
        # -----------------------------------------

        results = self.search_textbook(
            rewritten_question,
            top_k=5
        )

        # -----------------------------------------
        # Evaluate retrieval
        # -----------------------------------------

        evaluation = self.evaluator.evaluate(
            results
        )

        print(
            "\nBest distance:",
            evaluation["score"]
        )

        print(
            "Relevant:",
            evaluation["relevant"]
        )

        # -----------------------------------------
        # Textbook contains answer
        # -----------------------------------------

        if evaluation["relevant"]:

            print(
                "\nUsing textbook RAG..."
            )

            answer = (
                self.generator.generate_from_rag(
                    rewritten_question,
                    results,
                    conversation_history
                )
            )

            return {
                "answer": answer,
                "source": "textbook",
                "rewritten_question":
                    rewritten_question,
                "relevant": True
            }

        # -----------------------------------------
        # Textbook does not contain answer
        # -----------------------------------------

        print(
            "\nTextbook context insufficient."
        )

        print(
            "Using Groq fallback..."
        )

        answer = (
            self.generator.generate_general(
                question,
                conversation_history
            )
        )

        return {
            "answer": answer,
            "source": "groq",
            "rewritten_question":
                rewritten_question,
            "relevant": False
        }