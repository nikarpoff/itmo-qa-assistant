from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus
from qdrant_client.http.exceptions import UnexpectedResponse

from embeddings import Embedder


class EmbeddingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DatabaseError(Exception):
    def __init__(self, message):
        super().__init__(message)


class DatabaseWorker():
    def __init__(self, client: QdrantClient, embedder: Embedder, collection_name: str, size=1024):
        """
        Класс для работы с базой данных Qdrant
        :param client: клиент базы данных
        :param collection_name: имя коллекции
        :param embedder: объект класса Embedder для векторизации текста
        """
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.size = size
    
    def create_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection {self.collection_name} already exists")
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.size,
                    distance=Distance.COSINE
                )
            )

    def add_point(self, text, payload):
        """
        Векторизует текст и вставляет вектор в коллекцию
        :param text: текст для векторизации
        :param payload: дополнительные данные для вектора (title, text)
        """
        try:
            embedded_text = self.embedder.encode(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}")

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid4().hex),    # есть ID wiki, но мы будем использовать свой
                    vector=embedded_text,
                    payload=payload
                )
            ]
        )

        if operation_info.status != UpdateStatus.COMPLETED:
            raise DatabaseError(f"Failed to insert point: {operation_info}")

    def search(self, query, top_k=5):
        """
        Ищет в базе данных top_k наиболее похожих векторов на вектор запроса
        :param query: текст запроса
        :param top_k: количество возвращаемых результатов
        :return: список из top_k наиболее похожих векторов
        """
        try:
            embedded_query = self.embedder.encode(query, task="search_query")
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}")
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedded_query,
            limit=top_k,
            with_payload=True
        )

        return search_result
