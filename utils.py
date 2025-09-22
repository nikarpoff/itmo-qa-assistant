import os
import re
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
    Очищает текст от лишних конструкций.
    :param text: исходный текст
    :return: очищенный текст
    """
    text = text.strip().lower()

    # Удаляем специальные конструкции
    text = re.sub(r'__disambig__', '', text)
    text = re.sub(r'__notoc__', '', text)
    text = re.sub(r'__статическое_перенаправление__', '', text)
    text = re.sub(r'redirect.*', '', text)
    text = re.sub(r'перенаправление.*', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text) # {{...}}


    # Удаляем медиа-вставки (мини|..., thumb|..., File:...)
    text = re.sub(r'\|?мини\|.*?(\n|$)', ' ', text)
    text = re.sub(r'\|?thumb\|.*?(\n|$)', ' ', text)
    text = re.sub(r'\[\[file:.*?\]\]', '', text)


    # Удаляем wiki/HTML теги и ссылки
    text = re.sub(r'<ref.*?>.*?</ref>', '', text)
    text = re.sub(r'<.*?>', '', text) # HTML-теги
    text = re.sub(r'\[\[([^\]|]+)\|?([^\]]+)?\]\]', 
                  lambda m: m.group(2) if m.group(2) else m.group(1),
                  text)
    text = re.sub(r'\[.*?\]', '', text) # внешние ссылки

    # Приводим кавычки к нормальному виду
    text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')

    # Убираем шапки
    text = re.sub(r'^это статья о.*?\.', '', text, flags=re.MULTILINE)
    text = re.sub(r'^см\.? также.*', '', text, flags=re.MULTILINE)

    return text

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