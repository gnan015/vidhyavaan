from models.llm import GroqLLM
from agent.prompts import AGENT_SYSTEM_PROMPT


class AcademicAgent:

    def __init__(self):

        self.llm = GroqLLM()

    def decide(
        self,
        question: str,
        conversation_history=None
    ):

        # -----------------------------------------
        # Build conversation history
        # -----------------------------------------

        history_text = ""

        if conversation_history:

            for message in conversation_history:

                role = message["role"]
                content = message["content"]

                history_text += (
                    f"{role}: {content}\n"
                )

        # -----------------------------------------
        # Build routing prompt
        # -----------------------------------------

        prompt = f"""
{AGENT_SYSTEM_PROMPT}

CONVERSATION HISTORY:
---------------------
{history_text}
---------------------

STUDENT QUESTION:
{question}

Your task is to determine which route should
handle the student's request.

ROUTING RULES:

1. RAG
Use RAG when:
- The question is academic.
- The question is related to the student's textbooks.
- The question asks about an academic concept.
- The question asks for a definition or explanation.
- The question is a follow-up to a previous academic question.
- The question refers to something like:
  "the third one", "explain that", "what about the above",
  "explain the previous point", etc.

2. CALCULATOR
Use CALCULATOR when:
- The student asks for mathematical calculations.
- The student asks to add, subtract, multiply or divide numbers.
- The student asks for percentages.
- The student asks for arithmetic calculations.

Examples:
- What is 45 plus 67?
- Calculate 25 * 48.
- What is 15% of 200?
- Addition of 45 and 67.

3. GENERAL
Use GENERAL when:
- The question is clearly unrelated to academics.
- The student asks about general knowledge that is not
  related to their academic learning.
- The question is casual conversation.

IMPORTANT:
- Academic questions must go to RAG first.
- RAG will search the student's textbooks.
- If the textbook does not contain enough information,
  the system will later use Groq as a fallback.
- Do not use GENERAL merely because you are unsure
  whether the textbook contains the answer.

CONVERSATION HISTORY:
Use the conversation history to understand
follow-up questions.

For example:

Previous:
Student: What are the four conditions of deadlock?
AI: Mutual exclusion, hold and wait, no preemption,
and circular wait.

Current:
Student: Explain the third one.

The correct route is:
RAG

Return ONLY one of these:

RAG
CALCULATOR
GENERAL
"""

        # -----------------------------------------
        # Ask Groq for routing decision
        # -----------------------------------------

        decision = self.llm.generate(
            prompt
        )

        decision = decision.strip().upper()

        # -----------------------------------------
        # Handle accidental extra text
        # -----------------------------------------

        if "CALCULATOR" in decision:

            return "CALCULATOR"

        if "RAG" in decision:

            return "RAG"

        if "GENERAL" in decision:

            return "GENERAL"

        # -----------------------------------------
        # Safe default
        # -----------------------------------------

        # If the LLM gives an unexpected response,
        # send the question to RAG rather than directly
        # using general knowledge.

        return "RAG"