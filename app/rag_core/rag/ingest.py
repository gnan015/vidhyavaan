import pymupdf
from pathlib import Path

from models.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from a PDF page by page.

    Returns:
        List of dictionaries containing:
        - text
        - page number
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        if text.strip():

            pages.append({
                "text": text,
                "page": page_number + 1
            })

    document.close()

    return pages


def create_chunks(text, chunk_size=800, overlap=100):
    """
    Split text into overlapping chunks.

    chunk_size:
        Approximate number of characters in each chunk.

    overlap:
        Number of characters shared between consecutive chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_pdf(pdf_path: str):
    """
    Extract text from PDF and convert it into chunks.

    Each chunk contains:
        - text
        - book name
        - page number
    """

    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []

    book_name = Path(pdf_path).stem

    for page in pages:

        chunks = create_chunks(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "book": book_name,
                    "page": page["page"]
                }
            })

    return all_chunks


def main():

    pdf_path = "data/textbooks/Operating_System_Concepts_8th_EditionA4.pdf"

    print("Processing PDF...")

    documents = process_pdf(pdf_path)

    print(f"Total chunks created: {len(documents)}")

    print("\nGenerating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = embedding_model.encode(texts)

    print("\nStoring documents in ChromaDB...")

    vector_store = VectorStore()

    vector_store.add_documents(
        documents,
        embeddings
    )

    print("\n==============================")
    print("INGESTION COMPLETED")
    print("==============================")

    print(
        "Documents in database:",
        vector_store.count()
    )


if __name__ == "__main__":
    main()