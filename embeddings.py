import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging

# Отключаем предупреждения transformers
transformers_logging.set_verbosity_error()

def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


class LocalEmbedder():
    """
    Строит эмбеддинг для текста с помощью локальной модели
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")
        self.model = AutoModel.from_pretrained("ai-forever/ru-en-RoSBERTa")

    def encode(self, text, task="search_document"):
        """
        Строит эмбеддинг для текста с помощью локальной модели.
        :param text: текст для эмбеддинга
        :param task: префикс для текста (по умолчанию "search_document" - для улучшения кодирования ответов)
        :return: list-эмбеддинг
        """
        prefixed_text = f"{task}: {text}"

        tokenized_inputs = self.tokenizer(prefixed_text,
                                          max_length=512,
                                          padding=True,
                                          truncation=True,
                                          return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        embeddings = pool(
            outputs.last_hidden_state, 
            tokenized_inputs["attention_mask"],
            pooling_method="cls"
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings[0].tolist()
    