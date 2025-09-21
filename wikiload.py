import os
import time
import requests
import json

# Parsing library for MediaWiki markup
import mwparserfromhell

import config


def fetch_page_content(page_id: str):
    """
    Загружает текст статьи по её названию.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "pageids": page_id,
        "format": "json"
    }

    response = requests.get(config.WIKI_API_URL, params=params)
    response.raise_for_status()

    data = response.json()

    # Получаем текст статьи и удаляем вики-разметку
    text = data["query"]["pages"][str(page_id)]["revisions"][-1]["slots"]["main"]["*"]
    wikicode = mwparserfromhell.parse(text)

    return wikicode.strip_code()

def fetch_all_pages_content(pages: list, filename: str, report_times=10):
    """
    Загружает текст всех страниц из списка pages и сохраняет их в файл filename.
    """
    wiki_data = []
    total_pages = len(pages)
    report_interval = max(1, total_pages // report_times)

    for i, p in enumerate(pages, 1):
        page_id = p["pageid"]
        text = fetch_page_content(page_id)

        wiki_data.append({
            "id": page_id,
            "title": p["title"],
            "text": text
        })

        if i % report_interval == 0:
            print(f"\t\t\tFetched {i} pages...")

        time.sleep(0.5)  # не спамим

    # Набрали и загрузили pages_per_file -> выгружаем wiki_data в отдельный файл
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, indent=4, ensure_ascii=False)


def save_wiki(pages_per_file=5000, requests_limit=500):
    """
    Загружает и сохраняет содержимое всех страниц вики, разбивая на файлы по pages_per_file страниц.
    """
    if not os.path.exists(config.DATA_PATH):
        os.mkdir(config.DATA_PATH)
    
    print("Fetching wiki pages...")

    # Сначала получаем список всех страниц
    pages = []
    params = {
        "action": "query",
        "list": "allpages",         # запрос: все страницы
        "apnamespace": 0,           # только статьи
        "aplimit": requests_limit,  # максимальное число страниц за запрос
        "format": "json"
    }

    iteration = 0
    file_index = 0
    is_pages_remaining = True

    while is_pages_remaining:
        response = requests.get(config.WIKI_API_URL, params=params)
        response.raise_for_status()
        
        data = response.json()

        pages.extend(data["query"]["allpages"])

        iteration += 1
        print(f"\t Iteration {iteration}. Fetched {len(data['query']['allpages'])} titles.")

        # Если набралось pages_per_file страниц, то начинаем извлекать текст статей и сохраняем их в новый файл
        if len(pages) >= pages_per_file:
            print("\t\tFetching pages content...")

            filename = f"{config.DATA_PATH}/wiki_pages_{file_index}.json"
            fetch_all_pages_content(pages, filename)

            print(f"\t\tNext {pages_per_file} saved to {filename}")

            file_index += 1
            pages = []  # очищаем список страниц после сохранения

        # Проверяем, есть ли ещё страницы для загрузки
        if "continue" in data:
            params.update(data["continue"])  # обновляем параметры для следующего запроса
            time.sleep(0.5)  # не спамим
        else:
            is_pages_remaining = False  # все страницы загружены
    
    if pages:
        # Сохраняем оставшиеся страницы, если они есть
        print("\t\tFetching remaining pages content...")

        filename = f"{config.DATA_PATH}/wiki_pages_{file_index}.json"
        fetch_all_pages_content(pages, filename)

        print(f"\t\tRemaining {len(pages)} saved to {filename}")

    print("All done!")

save_wiki(pages_per_file=1000, requests_limit=500)
