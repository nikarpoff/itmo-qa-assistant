import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gigachat import GigaChat
from openai import OpenAI

import requests
from uuid import uuid4

import config


class InvalidAuthKeyException(Exception):
    def __init__(self, message):
        super().__init__(message)

class LLM:
    def answer(self, system_prompt: str, user_prompt: str) -> str:
        pass

class LocalLLM(LLM):
    def __init__(self, model_name="ai-forever/ruGPT-3.5-13B", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map="auto",
            # dtype="auto",
            # trust_remote_code=True
        )

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512
        )

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

class RemoteGigaChatLLM(LLM):
    def __init__(self, scope: str, authorization_key: str):
        self.auth_url = config.SBER_AUTH_URL
        self.model_url = config.GIGACHAT_API_URL
        self.scope = scope
        self.authorization_key = authorization_key

        self.giga = GigaChat(
            credentials=authorization_key,
            ca_bundle_file="russian_trusted_root_ca.cer"
        )

        self.access_key = self.generate_access_key()

    def generate_access_key(self) -> str:
        # payload = f"scope={self.scope}"

        # headers = {
        #   'Content-Type': 'application/x-www-form-urlencoded',
        #   'Accept': 'application/json',
        #   'RqUID': uuid4().hex,
        #   'Authorization': f"Basic {self.authorization_key}"
        # }

        # response = requests.request("POST", self.auth_url, headers=headers, data=payload, verify=False)
        
        # print(response)
        # response_json = response.json()

        # return response_json.get("access_token", "")
        return self.giga.get_token()


    def answer(self, system_prompt: str, user_prompt: str) -> str:
        # headers = {
        #   'Content-Type': 'application/json',
        #   'Accept': 'application/json',
        #   'Authorization': f"Bearer {self.authorization_key}"
        # }

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ]

        # payload={
        #     "model": "GigaChat",
        #     "messages": messages,
        #     "stream": False,
        #     "repetition_penalty": 1
        # }

        # response = requests.request("POST", self.model_url, headers=headers, data=payload)
        
        # if response.status_code == 200:
        #     response_json = response.json()
        #     return response_json["choices"][0]["message"]["content"]
        # elif response.status_code == 401:
        #     print("Exception 401 occuried! Authorization key was regenerated! Try to send query again...")
        #     self.last_access_key = self.get_access_key()
        #     raise InvalidAuthKeyException("Invalid authorization key. Maybe, key expired")
        # else:
        #     raise Exception(f"Request error occured: {response.json()}")
        response = self.giga.chat(system_prompt)

        return response.choices[0].message.content

class RemoteDeepseekLLM(LLM):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url=config.DEEPSEEK_API_URL)

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False
        )


        return response.choices[0].message.content

