"""
HuggingFace Inference API client for remote reference model (Qwen3-32B).

Requires: HF_TOKEN environment variable (huggingface.co/settings/tokens).
Falls back to Qwen2.5-72B-Instruct if Qwen3-32B is unavailable.

Usage:
  from src.models import hf_client as hf
  text = hf.generate("What is the capital of France?")
  obj  = hf.generate_json("Extract facts: ...", schema_hint="facts list")
"""
import os, json, re
from huggingface_hub import InferenceClient

_MODELS = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

_client = None
_active_model = None


def _get_client():
    global _client, _active_model
    if _client is not None:
        return _client, _active_model

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Run: export HF_TOKEN=<your_token>\n"
            "Get one at huggingface.co/settings/tokens"
        )

    c = InferenceClient(token=token)
    for model_id in _MODELS:
        try:
            r = c.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model=model_id, max_tokens=5
            )
            _ = r.choices[0].message.content  # smoke-test
            _client = c
            _active_model = model_id
            print(f"[hf_client] Using model: {model_id}")
            return _client, _active_model
        except Exception:
            continue

    raise RuntimeError(f"None of the candidate models are available: {_MODELS}")


def active_model():
    _, m = _get_client()
    return m


def generate(prompt, temperature=0, max_tokens=512, _retries=4):
    import time as _time
    client, model = _get_client()
    for attempt in range(_retries):
        try:
            r = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            text = r.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
            return text.strip()
        except Exception as e:
            if attempt == _retries - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [hf_client] Error: {e} — retrying in {wait}s…", flush=True)
            _time.sleep(wait)


def generate_json(prompt, temperature=0, max_tokens=600):
    full_prompt = prompt + "\nOutput JSON only, no extra text."
    raw = generate(full_prompt, temperature=temperature, max_tokens=max_tokens)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None
