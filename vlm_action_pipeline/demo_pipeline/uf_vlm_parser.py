import os, json
from typing import List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

UF_API_KEY  = #FILL IN
UF_BASE_URL = #FILL IN
UF_MODEL    = os.getenv("UF_MODEL", "llama-3.3-70b-instruct")

class UFVLMParser:
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.2):
        self.model = model_name or UF_MODEL
        self.client = OpenAI(api_key=UF_API_KEY, base_url=UF_BASE_URL)
        self.temperature = temperature

    @staticmethod
    def _ensure_single(messages: Union[list, str]) -> list:
        # messages must be a list[turn]; allow JSON string
        if isinstance(messages, str):
            messages = json.loads(messages)
        if not (isinstance(messages, list) and messages and isinstance(messages[0], dict)):
            raise ValueError("messages must be a single conversation: list[{'role','content'}, ...]")
        return messages

    @staticmethod
    def _ensure_batch(messages: Union[list, str]) -> List[list]:
        # batch must be list[list[turn]]
        if isinstance(messages, str):
            messages = json.loads(messages)
        if not (isinstance(messages, list) and messages and isinstance(messages[0], list)):
            raise ValueError("batch must be list of conversations: list[list[turn]]")
        return messages

    def query_llm(self, messages: Union[list, str], constraint_words: Optional[List[str]] = None) -> str:
        msgs = self._ensure_single(messages)
        if constraint_words:
            # light constraint: prepend to system
            system = {"role": "system",
                      "content": f"Only use the following terms if relevant: {', '.join(constraint_words)}"}
            msgs = [system] + msgs

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def query_llm_batch(self, messages: Union[list, str], constraint_words: Optional[List[str]] = None) -> List[str]:
        batch = self._ensure_batch(messages)
        out = []
        for conv in batch:
            out.append(self.query_llm(conv, constraint_words))
        return out
    
    def choose_from_enum(self, messages, choices, none_label="none"):
        # normalize shapes just like your other helpers
        msgs = self._ensure_single(messages)
        enum = list(dict.fromkeys([c.strip().lower() for c in choices + [none_label]]))

        schema = {
            "name": "action_choice",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": enum
                    }
                }
            },
            "strict": True
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs + [{
                    "role": "system",
                    "content": (
                        "Return a JSON object with a single field 'label' whose value is exactly "
                        "one of the allowed labels. Do not invent synonyms."
                    )
                }],
                temperature=0,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            obj = json.loads(resp.choices[0].message.content)
            label = obj.get("label", none_label)
            return label if label in enum else none_label
        except Exception:
            # Fallback (in case the backend doesn't support json_schema)
            # — Switch to ID voting to keep it strict.
            numbered = "\n".join(f"{i}: {c}" for i, c in enumerate(enum))
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content":
                        ("You are a strict classifier. Choose exactly one ID from the list. "
                         "If uncertain choose 0. Output ONLY the number.")},
                    *msgs,
                    {"role": "user", "content": f"Allowed actions:\n{numbered}\nAnswer:"},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\d+", text)
            idx = int(m.group()) if m else 0
            if not (0 <= idx < len(enum)):
                idx = 0
            return enum[idx]

