"""Provider-agnostic VLM client with retry, model rotation and fallback."""

import base64
import json
import os
import random
import time
import urllib.error
import urllib.request

POOLS = {
    "google": ["gemini-flash-latest", "gemini-3-flash-preview",
               "gemini-3.1-flash-lite", "gemini-flash-lite-latest"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-sonnet-4-5"],
}

MODELS = {p: v[0] for p, v in POOLS.items()}
KEYS = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"}

TIMEOUT = 120
RETRIES = 4


class ApiError(Exception):
    def __init__(self, code, body):
        self.code = code
        self.body = body
        super().__init__(f"{code}: {body[:300]}")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _post(url, payload, headers):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode(errors="replace"))


def _call(provider, model, img, prompt, max_tokens):
    key = os.environ[KEYS[provider]]

    if provider == "anthropic":
        out = _post(
            "https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": max_tokens,
             "messages": [{"role": "user", "content": [
                 {"type": "image", "source": {
                     "type": "base64", "media_type": "image/png",
                     "data": img}},
                 {"type": "text", "text": prompt}]}]},
            {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out["content"])

    if provider == "openai":
        out = _post(
            "https://api.openai.com/v1/chat/completions",
            {"model": model, "max_tokens": max_tokens,
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": prompt},
                 {"type": "image_url", "image_url": {
                     "url": f"data:image/png;base64,{img}"}}]}]},
            {"authorization": f"Bearer {key}"})
        msg = out["choices"][0]["message"]
        if msg.get("content") is None:
            return json.dumps({"_refusal": msg.get("refusal") or "empty"})
        return msg["content"]

    out = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}",
        {"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": img}}]}],
         "generationConfig": {"maxOutputTokens": max_tokens}},
        {})
    return "".join(p.get("text", "")
                   for p in out["candidates"][0]["content"]["parts"])


def _quota_exhausted(body):
    b = body.lower()
    return ("insufficient_quota" in b or "resource_exhausted" in b
            or "exceeded your current quota" in b
            or "billing" in b)


def ask(provider, image_path, prompt, model=None, max_tokens=2000,
        pool=None, verbose=False):
    img = _b64(image_path)
    models = [model] if model else (pool or POOLS[provider])
    last = None
    for m in models:
        delay = 2.0
        for attempt in range(RETRIES):
            try:
                return _call(provider, m, img, prompt, max_tokens)
            except ApiError as e:
                last = e
                if verbose:
                    print(f"      {m} -> {e.code} {e.body[:120]}")
                if e.code in (429, 503):
                    if _quota_exhausted(e.body):
                        break
                    time.sleep(delay + random.uniform(0, 1))
                    delay *= 2
                    continue
                if e.code in (400, 404):
                    break
                time.sleep(delay)
                delay *= 2
    raise last


def ask_any(image_path, prompt, providers=None, max_tokens=2000,
            verbose=False):
    providers = providers or [p for p in ("google", "openai", "anthropic")
                              if os.environ.get(KEYS[p], "").strip(". ")]
    last = None
    for p in providers:
        try:
            return p, ask(p, image_path, prompt, max_tokens=max_tokens,
                          verbose=verbose)
        except Exception as e:
            last = e
    raise last


DESCRIBE = ("Describe what you see in this image. What object or assembly "
            "is shown, and what state is it in? Be specific and concise.")


def diagnose():
    for p in ("google", "openai", "anthropic"):
        key = os.environ.get(KEYS[p], "").strip(". ")
        if not key:
            print(f"{p:10s} no key")
            continue
        print(f"{p:10s} key set ({len(key)} chars)")
        for m in POOLS[p]:
            try:
                _call(p, m, None, "hi", 20) if False else None
                url_ok = _probe(p, m)
                print(f"           {m:28s} {url_ok}")
            except ApiError as e:
                print(f"           {m:28s} {e.code} "
                      f"{'QUOTA' if _quota_exhausted(e.body) else 'other'}: "
                      f"{e.body[:110]}")
            except Exception as e:
                print(f"           {m:28s} {type(e).__name__}: {e}")


def _probe(provider, model):
    key = os.environ[KEYS[provider]]
    if provider == "anthropic":
        _post("https://api.anthropic.com/v1/messages",
              {"model": model, "max_tokens": 5,
               "messages": [{"role": "user", "content": "hi"}]},
              {"x-api-key": key, "anthropic-version": "2023-06-01"})
    elif provider == "openai":
        _post("https://api.openai.com/v1/chat/completions",
              {"model": model, "max_tokens": 5,
               "messages": [{"role": "user", "content": "hi"}]},
              {"authorization": f"Bearer {key}"})
    else:
        _post(f"https://generativelanguage.googleapis.com/v1beta/models/"
              f"{model}:generateContent?key={key}",
              {"contents": [{"parts": [{"text": "hi"}]}],
               "generationConfig": {"maxOutputTokens": 5}}, {})
    return "OK"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        diagnose()
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else "renders/pcb_normal.png"
        prov, text = ask_any(path, DESCRIBE, verbose=True)
        print(f"--- {prov}\n{text.strip()}")
