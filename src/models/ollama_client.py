import json, re, time, requests

OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 1200
MAX_RETRIES = 4   # retry transient drops (RemoteDisconnected / timeout / 5xx)

# Qwen3 routes constrained JSON to the "thinking" field and leaves "response"
# empty; non-JSON calls put thinking there too and only populate "response"
# once thinking finishes (needs large num_predict budget). Both helpers handle
# this transparently.

def _post(body):
    """POST with retry-and-backoff on transient errors; raise loudly if all
    attempts fail (never silently degrade — a swallowed failure would corrupt
    op-recall by masquerading as NOOP)."""
    body.setdefault("keep_alive", "30m")   # pin model resident between calls
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OLLAMA_URL, json=body, timeout=TIMEOUT)
            if r.status_code >= 500:
                raise requests.exceptions.HTTPError(f"{r.status_code} {r.text[:120]}")
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)   # 1s, 2s, 4s backoff
    raise RuntimeError(f"Ollama call failed after {MAX_RETRIES} attempts: {last}")

def generate(model, prompt, temperature=0, think=True):
    body = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 800},
    }
    if not think:
        body["think"] = False
    data = _post(body)
    text = data["response"].strip()
    if not text:
        # thinking model: extract answer from end of thinking block
        thinking = data.get("thinking", "")
        # last non-empty line is typically the final answer
        lines = [l.strip() for l in thinking.splitlines() if l.strip()]
        text = lines[-1] if lines else ""
    return text

def generate_json(model, prompt, temperature=0, think=True, max_tokens=600, schema=None):
    # schema (a JSON Schema dict) constrains output structure via Ollama's
    # structured-output support; falls back to free "json" mode if None.
    body = {
        "model": model, "prompt": prompt, "stream": False,
        "format": schema if schema is not None else "json",
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 2048},
    }
    if not think:
        body["think"] = False
    data = _post(body)
    text = data["response"].strip()
    if not text:
        # Qwen3: constrained JSON lands in "thinking" field
        text = data.get("thinking", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try extracting the first {...} block
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None
