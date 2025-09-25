import os
from dotenv import load_dotenv

import llm
import utils
from promts import get_system_prompt


class RAGAssistant:
    def __init__(self, model_type="deepseek-remote"):
        load_dotenv()
        self.model = None

        if model_type == "local":
            self.model = llm.LocalLLM()
        elif model_type == "gigachat-remote":
            scope = os.getenv("SBER_SCOPE")
            authorization_key = os.getenv("SBER_API_KEY")

            if not scope or not authorization_key:
                raise ValueError("API URL, scope, or authorization key is not set in environment variables.")

            self.model = llm.RemoteGigaChatLLM(
                scope=scope,
                authorization_key=authorization_key
            )
        elif model_type == "deepseek-remote":
            api_key = os.getenv("DEEPSEEK_API_KEY")

            self.model = llm.RemoteDeepseekLLM(
                api_key=api_key
            )
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented yet.")

        self.db_worker = utils.get_db_worker()

    def generate_answer(self, question):
        results = self.db_worker.search(question, top_k=5)
        context_documents = []

        for r in results:
            context_documents.append(r.payload.get("content", ""))

        context = "\n".join(context_documents)
        system_prompt = get_system_prompt(question, context, role="default")

        answer = ""

        try:
            answer = self.model.answer(system_prompt, question)
        except llm.InvalidAuthKeyException:
            # Try again with regenerated key
            answer = self.model.answer(system_prompt, question)
        except Exception as e:
            print("Exception: ", e)

        return answer


if __name__ == "__main__":
    assistant = RAGAssistant(model_type="gigachat-remote")
    question = "Что такое исполнитель желаний?"
    answer = assistant.generate_answer(question)
    print(f"Question: {question}\nAnswer: {answer}")