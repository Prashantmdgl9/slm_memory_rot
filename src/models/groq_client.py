"""
Groq Inference API client — drop-in replacement for hf_client.

Requires: GROQ_API_KEY environment variable (console.groq.com)

Default model: llama-3.3-70b-versatile (free tier, no credit cap)
"""
import os, json, re, time
from groq import Groq

_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-70b-versatile",
]

_client = None
_active_model = None


def _get_client():
    global _client, _active_model
    if _client is not None:
        return _client, _active_model

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at console.groq.com/keys\n"
            "Then run: export GROQ_API_KEY=gsk_..."
        )

    c = Groq(api_key=key)
    for model_id in _MODELS:
        try:
            r = c.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model=model_id, max_tokens=5
            )
            _ = r.choices[0].message.content
            _client = c
            _active_model = model_id
            print(f"[groq_client] Using model: {model_id}")
            return _client, _active_model
        except Exception:
            continue

    raise RuntimeError(f"None of the Groq models are available: {_MODELS}")


def active_model():
    _, m = _get_client()
    return m


def _parse_retry_after(err_str):
    """Extract wait seconds from Groq rate-limit message."""
    import re
    s = str(err_str)
    # e.g. "try again in 1h0m42.624s"
    m = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s', s)
    if m:
        h = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        secs = float(m.group(3) or 0)
        return h * 3600 + mins * 60 + secs + 10
    return None


def generate(prompt, temperature=0, max_tokens=512, _retries=8):
    client, model = _get_client()
    for attempt in range(_retries):
        try:
            r = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == _retries - 1:
                raise
            wait = _parse_retry_after(e) or (15 * (attempt + 1))
            print(f"  [groq_client] Error: {e} — retrying in {wait:.0f}s…", flush=True)
            time.sleep(wait)


def generate_json(prompt, temperature=0, max_tokens=600):
    raw = generate(prompt + "\nOutput JSON only, no extra text.",
                   temperature=temperature, max_tokens=max_tokens)
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
