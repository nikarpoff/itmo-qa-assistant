import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
import database
from embeddings import LocalEmbedder


load_dotenv()

DATABASE_CONNECTION_URL = os.getenv("DATABASE_CONNECTION_URL")


def get_db_worker(embedder_type="local"):
    """
    Создает и возвращает объект DatabaseWorker
    :param embedder: тип эмбеддера (по умолчанию "local")
    """
    client = QdrantClient(DATABASE_CONNECTION_URL)
    embedder = None

    if embedder_type == "local":
        embedder = LocalEmbedder()
    else:
        raise Exception(f"Unknown embedder type: {embedder}")

    db_worker = database.DatabaseWorker(client,
                                        embedder,
                                        config.COLLECTION_NAME,
                                        config.VECTOR_SIZE
                                        )
    return db_worker

def clear_text(text: str) -> str:
    """
    Очищает текст от лишних пробелов и символов переноса строки.
    :param text: исходный текст
    :return: очищенный текст
    """
    return text.strip().lower()

def split_by_chunks(text: str, chunk_size=500, overlap=75) -> list:
    """
    Разбивает текст на чанки заданного размера с заданным перекрытием.
    :param text: исходный текст
    :param chunk_size: размер чанка
    :param overlap: размер перекрытия между чанками
    :return: список чанков
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n"]
    )
    
    return splitter.split_text(text)