import os
import json

import config
import utils
import database


def is_valid_page(page: dict) -> bool:
    """
    Проверяет, является ли страница валидной для добавления в БД.
    :param page: страница
    :return: True, если страница валидна, False иначе
    """
    title = page.get("title", "").lower()
    text = page.get("text", "").lower()

    # Проверки на валидность страницы
    if not title or not text:
        return False

    # Отклоняем служебные страницы
    if title.startswith("категория:") or title.startswith("файл:") or title.startswith("обсуждение:"):
        return False

    if len(text) < config.MIN_PAGE_LENGTH:
        return False

    # Такие статьи не содержат полезной информации
    if "перенаправление" in text or "redirect" in text:
        return False

    return True


def write_pages_into_db(filename, db_worker: database.DatabaseWorker):
    """
    Записывает в БД все страницы из файла filename
    """
    with open(filename, 'r', encoding="utf-8") as file:
        pages = json.load(file)

        for page in pages:
            if not is_valid_page(page):
                print(f"\t Skipping invalid page: {page.get('title')} (id={page.get('id')})")
                continue

            id = page.get("id")
            title = page.get("title")
            text = page.get("text")
            
            # Очистим текст от мусора
            cleared_text = utils.clear_text(text)

            # Делим на чанки
            chunked_text = utils.split_by_chunks(cleared_text, db_worker.embedder.tokenizer, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

            print(f"\t Writing page: {title} (id={id}), {len(chunked_text)} chunks")

            for i, chunk in enumerate(chunked_text):
                payload = {
                    "content": chunk,    # в payload сохраняем весь текст статьи
                    "full_text": text,
                    "metadata": {
                        "wiki_id": id,
                        "chunk": i,
                        "title": title
                    }
                }

                try:
                    db_worker.add_point(chunk, payload)
                    print(f"\t\t Added point: chunk {i}")
                except database.DatabaseError | database.EmbeddingError as e:
                    print(f"\t\t Error adding point {title} (id={id}) chunk {i}: {e}")

def write_wiki_into_db(db_worker: database.DatabaseWorker):
    """
    Записывает в БД все страницы из всех батчей в директории config.DATA_PATH
    """
    if not os.path.exists(config.DATA_PATH):
        raise Exception("Data package is empty! Load data first!")

    # Пишем все страницы, пока они есть
    file_index = 0
    while True:
        filename = f"{config.DATA_PATH}/wiki_pages_{file_index}.json"

        if not os.path.exists(filename):
            break
        
        print(f"Uploading pages batch {file_index}")
        write_pages_into_db(filename, db_worker)

        file_index += 1

    print("All done!")

if __name__ == "__main__":
    db_worker = utils.get_db_worker()
    db_worker.create_collection()

    write_wiki_into_db(db_worker)
