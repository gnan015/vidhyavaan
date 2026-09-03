from rag.retriever import Retriever
from rag.generator import RAGGenerator
from rag.evaluator import RAGEvaluator
from rag.query_rewriter import QueryRewriter

from agent.conversation import ConversationMemory


def main():

    retriever = Retriever()

    generator = RAGGenerator()

    evaluator = RAGEvaluator(
        threshold=1.2
    )

    rewriter = QueryRewriter()

    memory = ConversationMemory()

    print("\n================================")
    print("        SIGNALMINDS AI")
    print("================================")

    print("Ask your academic questions.")
    print("Type 'exit' to stop.")
    print("Type 'clear' to start a new conversation.")

    while True:

        question = input("\nStudent: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if question.lower() == "clear":

            memory.clear()

            print("\nConversation memory cleared.")

            continue

        # -----------------------------------------
        # Get previous conversation
        # -----------------------------------------

        previous_history = memory.get_history().copy()

        # -----------------------------------------
        # Rewrite question for RAG
        # -----------------------------------------

        rewritten_question = rewriter.rewrite(
            question,
            previous_history
        )

        print("\nOriginal question:")
        print(question)

        print("\nSearch query:")
        print(rewritten_question)

        # -----------------------------------------
        # Save original question
        # -----------------------------------------

        memory.add_user_message(question)

        # -----------------------------------------
        # Search ChromaDB
        # -----------------------------------------

        print("\nSearching textbook...")

        results = retriever.search(
            rewritten_question,
            top_k=5
        )

        # -----------------------------------------
        # Evaluate RAG results
        # -----------------------------------------

        evaluation = evaluator.evaluate(results)

        print("\nBest distance:", evaluation["score"])
        print("Relevant:", evaluation["relevant"])

        # -----------------------------------------
        # Generate answer
        # -----------------------------------------

        if evaluation["relevant"]:

            print("\nUsing textbook RAG...")

            answer = generator.generate_from_rag(
                rewritten_question,
                results,
                previous_history
            )

        else:

            print("\nTextbook context insufficient.")

            print("Using Groq fallback...")

            answer = generator.generate_general(
                question,
                previous_history
            )

        # -----------------------------------------
        # Display answer
        # -----------------------------------------

        print("\nAI:")
        print(answer)

        # -----------------------------------------
        # Save AI response
        # -----------------------------------------

        memory.add_assistant_message(answer)


if __name__ == "__main__":
    main()