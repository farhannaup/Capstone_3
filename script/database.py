import os
import logging
import pandas as pd
from uuid import uuid4
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# =====================
# ENV & LOGGING
# =====================

load_dotenv()
logging.basicConfig(level=logging.INFO)

# =====================
# QDRANT CLIENT
# =====================

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# =====================
# EMBEDDING MODEL
# =====================

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# =====================
# TEXT SPLITTER (CHUNKING)
# =====================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
)

# =====================
# LOAD & CHUNK DATA
# =====================

def load_imdb(csv_path: str, limit: int = 200) -> list[Document]:
    logging.info("Loading IMDB dataset...")
    df = pd.read_csv(csv_path).head(limit)

    documents = []

    for idx, row in df.iterrows():
        overview = str(row.get("Overview", "")).strip()
        if not overview:
            continue

        chunks = text_splitter.split_text(overview)

        for chunk_id, chunk in enumerate(chunks):
            content = f"""
Title: {row.get('Series_Title')}
Year: {row.get('Released_Year')}
Genre: {row.get('Genre')}
Director: {row.get('Director')}
IMDB Rating: {row.get('IMDB_Rating')}

Overview:
{chunk}
""".strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "title": row.get("Series_Title"),
                        "year": row.get("Released_Year"),
                        "genre": row.get("Genre"),
                        "director": row.get("Director"),
                        "rating": row.get("IMDB_Rating"),
                        "chunk_id": chunk_id,
                        "row_index": idx,
                        "source": "imdb_movies.csv",
                    },
                )
            )

    logging.info(f"Total documents created: {len(documents)}")
    return documents

# =====================
# INGESTION PIPELINE
# =====================

def main():
    logging.info("Starting IMDB ingestion pipeline...")

    documents = load_imdb(
        csv_path="dataset/imdb_movies.csv",
        limit=200,
    )

    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    collections = [c.name for c in qdrant_client.get_collections().collections]

    if collection_name not in collections:
        logging.info("Creating Qdrant collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    vector_store.add_documents(
        documents=documents,
        ids=[str(uuid4()) for _ in documents],
    )

    logging.info("IMDB ingestion completed successfully.")

if __name__ == "__main__":
    main()
