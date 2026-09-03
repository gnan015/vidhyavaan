from agent.agent import AcademicAgent
from agent.conversation import ConversationMemory

from rag.generator import RAGGenerator

from tools.tool_manager import ToolManager


def main():

    # =========================================
    # Initialize components
    # =========================================

    agent = AcademicAgent()

    memory = ConversationMemory()

    generator = RAGGenerator()

    tool_manager = ToolManager()

    # =========================================
    # Application header
    # =========================================

    print("\n================================")
    print("        SIGNALMINDS AI")
    print("================================")

    print("Ask your academic questions.")
    print("Type 'exit' to stop.")
    print("Type 'clear' to clear conversation.")

    # =========================================
    # Main conversation loop
    # =========================================

    while True:

        question = input("\nStudent: ").strip()

        # -----------------------------------------
        # Ignore empty input
        # -----------------------------------------

        if not question:
            continue

        # -----------------------------------------
        # Exit
        # -----------------------------------------

        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        # -----------------------------------------
        # Clear conversation
        # -----------------------------------------

        if question.lower() == "clear":

            memory.clear()

            print("\nConversation memory cleared.")

            continue

        # =========================================
        # Previous conversation
        # =========================================

        previous_history = (
            memory.get_history().copy()
        )

        # =========================================
        # AGENT
        # =========================================

        decision = agent.decide(
            question,
            previous_history
        )

        print("\nAgent decision:")
        print(decision)

        # -----------------------------------------
        # Save user question
        # -----------------------------------------

        memory.add_user_message(
            question
        )

        # =========================================
        # CALCULATOR
        # =========================================

        if decision == "CALCULATOR":

            print("\nCalculator selected.")

            result = tool_manager.calculate(
                question
            )

            if result is not None:

                answer = (
                    f"The answer is {result}."
                )

            else:

                answer = (
                    "Sorry, I could not understand "
                    "the mathematical expression."
                )

        # =========================================
        # GENERAL
        # =========================================

        elif decision == "GENERAL":

            print(
                "\nUsing Groq general knowledge..."
            )

            answer = generator.generate_general(
                question,
                previous_history
            )

        # =========================================
        # RAG
        # =========================================

        elif decision == "RAG":

            result = tool_manager.answer_from_textbook(
                question,
                previous_history
            )

            answer = result["answer"]

            # -------------------------------------
            # Display source
            # -------------------------------------

            if result["source"] == "textbook":

                print(
                    "\nAnswer generated from textbook."
                )

            else:

                print(
                    "\nAnswer generated using Groq fallback."
                )

        # =========================================
        # UNKNOWN DECISION
        # =========================================

        else:

            answer = (
                "Sorry, I could not determine "
                "how to handle your question."
            )

        # =========================================
        # FINAL ANSWER
        # =========================================

        print("\nAI:")
        print(answer)

        # =========================================
        # Save AI response
        # =========================================

        memory.add_assistant_message(
            answer
        )


if __name__ == "__main__":
    main()