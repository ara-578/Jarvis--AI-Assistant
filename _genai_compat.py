"""
_genai_compat.py — thin shim that lets the rest of the codebase keep using
the old `model = genai.GenerativeModel(...)` / `model.generate_content(...)`
style, while actually talking to the new `google.genai` SDK under the hood.

The legacy `google.generativeai` package is deprecated and will eventually
stop working, so every caller in this project should import `get_model`
from here instead of doing `import google.generativeai as genai` directly.
"""

import json
from pathlib import Path

from google import genai
from google.genai import types

_client = None


def _get_api_key() -> str:
    config_path = Path(__file__).resolve().parent / "config" / "api_keys.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=_get_api_key())
    return _client


class _ModelShim:
    """Mimics the old google.generativeai GenerativeModel interface."""

    def __init__(self, model_name: str, system_instruction: str | None = None):
        # Old SDK model names sometimes came prefixed with "models/"; the
        # new SDK wants the bare name.
        self.model_name = model_name.removeprefix("models/")
        self.system_instruction = system_instruction

    def generate_content(self, contents):
        config = None
        if self.system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction
            )
        return _get_client().models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )


def get_model(model_name: str, system_instruction: str | None = None) -> _ModelShim:
    """Drop-in replacement for genai.GenerativeModel(model_name, system_instruction=...)."""
    return _ModelShim(model_name, system_instruction=system_instruction)
