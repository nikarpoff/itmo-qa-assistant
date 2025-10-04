import os
from dotenv import load_dotenv
from argparse import ArgumentParser

import llm
import utils
from promts import get_system_prompt


def cli_arguments_preprocess() -> str:
    parser = ArgumentParser(description="A script for asking LLM a question about the lore of the game S.T.A.L.K.E.R.")

    parser.add_argument('question',
                        help='Question for LLM')

    parser.add_argument("--model_type", required=True,
                      help="Type of model: gigachat-remote/deepseek-remote/local")
    
    parser.add_argument("--model_name", required=False, default=None,
                      help="Name of model like GigaChat-Pro or Hugging Face model (in local model case)")
    
    parser.add_argument("--role", required=False, default="default",
                      help="Role of the model (default/bandit)")

    args = parser.parse_args()

    return args.question, args.model_type, args.model_name, args.role


class RAGAssistant:
    def __init__(self, model_type="deepseek-remote", model_name=None):
        load_dotenv()
        self.model = None

        if model_type == "local":
            if not model_name:
                model_name = "ai-forever/ruGPT-3.5-13B"

            self.model = llm.LocalLLM(model_name)
        elif model_type == "gigachat-remote":
            scope = os.getenv("SBER_SCOPE")
            authorization_key = os.getenv("SBER_API_KEY")

            if not scope or not authorization_key:
                raise ValueError("API URL, scope, or authorization key is not set in environment variables.")

            if not model_name:
                model_name = "GigaChat"
            
            self.model = llm.RemoteGigaChatLLM(
                model_name=model_name,
                scope=scope,
                authorization_key=authorization_key
            )
        elif model_type == "deepseek-remote":
            api_key = os.getenv("DEEPSEEK_API_KEY")

            self.model = llm.RemoteDeepseekLLM(
                api_key=api_key
            )
        else:
            raise ValueError(f"Model type {model_type} not realized yet")

        self.db_worker = utils.get_db_worker()

    def generate_answer(self, question, role="default"):
        results = self.db_worker.search(question, top_k=5)
        context_documents = []

        for r in results:
            context_documents.append(
                f"{r.payload.get('title', '')}: {r.payload.get('content', '')}"
            )

        context = "\n".join(context_documents)
        system_prompt = get_system_prompt(question, context, role)

        answer = ""

        try:
            answer = self.model.answer(system_prompt, question)
        except llm.InvalidAuthKeyException:
            # Try again with regenerated key
            answer = self.model.answer(system_prompt, question)
        except Exception as e:
            print("Exception: ", e)

        return answer


def main(question: str, model_type: str, model_name=None, role="default"):
    assistant = RAGAssistant(model_type=model_type, model_name=model_name)
    answer = assistant.generate_answer(question, role)
    print(f"Question: {question}\nAnswer: {answer}")

if __name__ == "__main__":
    question, model_type, model_name, role = cli_arguments_preprocess()
    main(question, model_type, model_name, role)
