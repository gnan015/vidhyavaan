from groq import Groq
import os

from config import GROQ_API_KEY


class GroqLLM:

    def __init__(self):

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set in .env"
            )

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        # This is available to standard Groq developer accounts and is fast
        # enough for phone replies. Deployments can override it in .env.
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    # --------------------------------------------------
    # Normal LLM generation
    # --------------------------------------------------

    def generate(
        self,
        prompt: str,
        conversation_history=None,
        system_prompt: str | None = None,
    ):

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content

    # --------------------------------------------------
    # Query Rewriting
    # --------------------------------------------------

    def rewrite_query(
        self,
        question: str,
        conversation_history=None
    ):

        history_text = ""

        if conversation_history:

            for message in conversation_history:

                role = message["role"]
                content = message["content"]

                history_text += (
                    f"{role}: {content}\n"
                )

        prompt = f"""
You are a query rewriting assistant for an academic
Retrieval-Augmented Generation (RAG) system.

Your task is to convert the student's latest question
into a clear, standalone search query that can be used
to search academic textbooks.

CONVERSATION HISTORY:
---------------------
{history_text}
---------------------

LATEST STUDENT QUESTION:
{question}

RULES:

1. Use the conversation history to understand the context.

2. Resolve references such as:
   - "it"
   - "that"
   - "this"
   - "the second one"
   - "the previous topic"
   - "explain that"
   - "give another example"
   - "why is it required"

3. Preserve the student's original intent.

4. Make the rewritten query self-contained.

5. Do NOT answer the student's question.

6. Return ONLY the rewritten search query.

EXAMPLE:

Conversation history:

Student:
What are the four necessary conditions for deadlock?

Assistant:
The four conditions are mutual exclusion,
hold and wait, no preemption and circular wait.

Latest student question:
Explain the second one.

Rewritten query:
Explain the hold and wait condition of deadlock
in operating systems.
"""

        return self.generate(prompt)
