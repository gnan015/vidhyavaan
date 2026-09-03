from pathlib import Path
import os

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Project root directory
BASE_DIR = Path(__file__).resolve().parent


# Data directories
TEXTBOOK_DIR = BASE_DIR / "data" / "textbooks"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# RAG
COLLECTION_NAME = "academic_documents"

# Number of chunks retrieved for each question
TOP_K = 5


# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"