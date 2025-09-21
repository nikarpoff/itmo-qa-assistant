from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        pass

    def encode(self, text, task="search_document"):
        pass

class LocalEmbedder(Embedder):
    """
    Строит эмбеддинг для текста с помощью локальной модели
    """
    def __init__(self):
        self.model = SentenceTransformer('ai-forever/ru-en-RoSBERTa')

    def encode(self, text, task="search_document"):
        """
        Строит эмбеддинг для текста с помощью локальной модели.
        :param text: текст для эмбеддинга
        :param task: префикс для текста (по умолчанию "search_document" - для улучшения кодирования ответов)
        :return: list-эмбеддинг
        """
        prefixed_text = f"{task}: {text}"

        return self.model.encode(
            prefixed_text,
            normize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=True
        ).tolist()
    