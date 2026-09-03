from agent.agent import AcademicAgent
from agent.conversation import ConversationMemory


def main():

    agent = AcademicAgent()

    memory = ConversationMemory()

    print("\n================================")
    print("       SIGNALMINDS AGENT")
    print("================================")

    print("Type 'exit' to stop.")

    while True:

        question = input("\nStudent: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        history = memory.get_history().copy()

        decision = agent.decide(
            question,
            history
        )

        print("\nAgent decision:")
        print(decision)

        memory.add_user_message(question)

        # Temporary response so that the agent
        # has conversational context during testing.
        memory.add_assistant_message(
            f"The system classified the request as {decision}."
        )


if __name__ == "__main__":
    main()