import os
import json

import config
import utils
import database


def write_pages_into_db(filename, db_worker: database.DatabaseWorker):
    """
    Записывает в БД все страницы из файла filename
    """
    with open(filename, 'r', encoding="utf-8") as file:
        pages = json.load(file)

        for page in pages:
            id = page.get("id")
            title = page.get("title")
            text = page.get("text")

            payload = {
                "content": text,
                "metadata": {
                    "wiki_id": id,
                    "title": title
                }
            }

            try:
                db_worker.add_point(text, payload)
            except database.DatabaseError | database.EmbeddingError as e:
                print(f"Error adding point {title} (id={id}): {e}")

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

    write_wiki_into_db()
