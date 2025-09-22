import utils
import database

if __name__ == "__main__":
    db_worker = utils.get_db_worker()
    query = "что такое исполнитель желаний?"

    try:
        results = db_worker.search(query)
        print(f"Query: {query}")

        for i, res in enumerate(results, 1):
            title = res.payload.get("metadata", {}).get("title", "No Title")
            text = res.payload.get("content", "No Content")
            score = res.score

            print(f"\nResult {i}:")
            print(f"Title: {title}")
            print(f"Score: {score:.4f}")
            print(f"Content: {text}...")  # Print first 200 characters of content

    except database.EmbeddingError as e:
        print(f"Embedding error: {e}")
