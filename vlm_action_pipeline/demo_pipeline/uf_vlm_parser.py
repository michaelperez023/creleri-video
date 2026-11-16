"""Client for the LLaMA-3.3 chat endpoint behind llama_3p3_api.py.

Any OpenAI-compatible server works. It is configured with LLM_BASE_URL, LLM_API_KEY and
LLM_MODEL, taken from the environment or a .env file. When LLM_BASE_URL is unset, the older
UF_BASE_URL, UF_API_KEY and UF_MODEL are read instead.
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-instruct"


class UFVLMParser:
    def __init__(self, model_name: str | None = None, temperature: float = 0.2):
        # Read the settings as one set so a legacy key is never sent to a new endpoint.
        prefix = "LLM" if os.getenv("LLM_BASE_URL") else "UF"
        base_url = os.getenv(f"{prefix}_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "No LLM endpoint configured: set LLM_BASE_URL (plus LLM_API_KEY and LLM_MODEL "
                "as needed) to an OpenAI-compatible server, in the environment or a .env file."
            )
        self.model = model_name or os.getenv(f"{prefix}_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        # Always pass a key: given None, the SDK falls back to OPENAI_API_KEY and would send
        # that key to this endpoint.
        self.client = OpenAI(
            api_key=os.getenv(f"{prefix}_API_KEY") or "",
            base_url=base_url,
            timeout=60,
            max_retries=2,
        )

    def query_llm(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    def choose_from_enum(
        self, messages: list[dict], choices: list[str], none_label: str = "none"
    ) -> str:
        """Pick one of `choices` for the conversation, or `none_label` when unsure."""
        none_label = none_label.strip().lower()
        enum = list(dict.fromkeys(c.strip().lower() for c in [*choices, none_label]))

        schema = {
            "name": "action_choice",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {"label": {"type": "string", "enum": enum}},
            },
            "strict": True,
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "Return a JSON object with a single field 'label' whose value is "
                            "exactly one of the allowed labels. Do not invent synonyms."
                        ),
                    },
                ],
                temperature=0,
                max_tokens=64,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            label = json.loads(resp.choices[0].message.content).get("label", none_label)
            return label if label in enum else none_label
        except Exception as e:
            # Not every backend supports json_schema; fall back to picking a numbered ID.
            logger.warning("Structured label request failed (%s); using numbered fallback", e)

        none_idx = enum.index(none_label)
        numbered = "\n".join(f"{i}: {c}" for i, c in enumerate(enum))
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=16,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict classifier. Choose exactly one ID from the list. "
                        f"If uncertain choose {none_idx}. Output ONLY the number."
                    ),
                },
                *messages,
                {"role": "user", "content": f"Allowed actions:\n{numbered}\nAnswer:"},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        if answer in enum:
            return answer
        m = re.search(r"\d+", answer)
        idx = int(m.group()) if m else none_idx
        return enum[idx] if idx < len(enum) else none_label
