from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from threading import Lock

from agent.agent import AcademicAgent
from agent.conversation import ConversationMemory
from rag.generator import RAGGenerator
from tools.tool_manager import ToolManager


app = FastAPI(
    title="SignalMinds AI",
    description="AI-powered academic voice assistant backend",
    version="1.0.0"
)


# --------------------------------------------------
# Shared components
# --------------------------------------------------

agent = AcademicAgent()
generator = RAGGenerator()
tool_manager = ToolManager()


# --------------------------------------------------
# User session storage
# --------------------------------------------------

sessions = {}

# Protects creation/access of sessions
sessions_lock = Lock()


def get_session(session_id: str):
    """
    Get an existing user session.

    If the session does not exist, create it dynamically.
    """

    with sessions_lock:

        if session_id not in sessions:
            sessions[session_id] = ConversationMemory()

        return sessions[session_id]


# --------------------------------------------------
# Request / Response models
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    session_id: str
    answer: str
    route: str
    source: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "service": "SignalMinds AI",
        "status": "running"
    }


@app.get("/health")
def health():

    with sessions_lock:
        active_sessions = len(sessions)

    return {
        "status": "running",
        "service": "SignalMinds AI",
        "active_sessions": active_sessions
    }


# --------------------------------------------------
# Ask endpoint
# --------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    # ----------------------------------------------
    # 1. Create or retrieve user session
    # ----------------------------------------------

    session_id = request.session_id

    if not session_id:
        session_id = str(uuid4())

    memory = get_session(session_id)


    # ----------------------------------------------
    # 2. Get previous conversation
    # ----------------------------------------------

    previous_history = memory.get_history().copy()


    # ----------------------------------------------
    # 3. Decide which route to use
    # ----------------------------------------------

    decision = agent.decide(
        request.question,
        previous_history
    )


    # ----------------------------------------------
    # 4. Store student's question
    # ----------------------------------------------

    memory.add_user_message(request.question)


    # ----------------------------------------------
    # 5. Process request
    # ----------------------------------------------

    if decision == "CALCULATOR":

        result = tool_manager.calculate(
            request.question
        )

        if result is not None:

            answer = f"The answer is {result}."

        else:

            answer = (
                "Sorry, I could not understand "
                "the mathematical expression."
            )

        source = "calculator"


    elif decision == "GENERAL":

        answer = generator.generate_general(
            request.question,
            previous_history
        )

        source = "groq"


    elif decision == "RAG":

        result = tool_manager.answer_from_textbook(
            request.question,
            previous_history
        )

        answer = result["answer"]

        if result["source"] == "textbook":

            source = "textbook"

        else:

            source = "groq"


    else:

        answer = (
            "Sorry, I could not determine "
            "how to handle your question."
        )

        source = "unknown"


    # ----------------------------------------------
    # 6. Store AI response
    # ----------------------------------------------

    memory.add_assistant_message(answer)


    # ----------------------------------------------
    # 7. Return response
    # ----------------------------------------------

    return AskResponse(
        session_id=session_id,
        answer=answer,
        route=decision,
        source=source
    )


# --------------------------------------------------
# Run directly
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )