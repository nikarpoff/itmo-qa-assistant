import utils
import database
from argparse import ArgumentParser


def cli_arguments_preprocess() -> str:
    parser = ArgumentParser(description="A script for search related with query documents in database")

    parser.add_argument('query', help='Query that will be encoded and searched for')

    args = parser.parse_args()

    return args.query

if __name__ == "__main__":
    db_worker = utils.get_db_worker()
    query = cli_arguments_preprocess()

    try:
        results = db_worker.search(query.lower())
        print(f"Query: {query}")

        for i, res in enumerate(results, 1):
            title = res.payload.get("metadata", {}).get("title", "No Title")
            text = res.payload.get("content", "No Content")
            score = res.score

            print(f"\n\nResult {i}:")
            print(f"Title: {title}")
            print(f"Score: {score:.4f}")
            print(f"Content: {text[:200]}...")  # Print first 200 characters of content

    except database.EmbeddingError as e:
        print(f"Embedding error: {e}")
