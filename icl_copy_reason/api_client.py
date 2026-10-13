"""One OpenAI-compatible chat client for every generator / verifier / auditor call.

Inputs : ApiSettings (ICR_API_BASE, ICR_API_KEY, model names) and a list of chat messages.
Outputs: the assistant text (and usage counters), with retry + exponential back-off on 429 / 5xx.
The package never prints or logs the key. Requires the optional dependency `openai` (pip install -e ".[api]").
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

from .config import ApiSettings


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion


class ChatClient:
    """Thread-safe wrapper around an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, settings: ApiSettings | None = None, retries: int = 6):
        self.settings = settings or ApiSettings.from_env()
        if not self.settings.base_url or not self.settings.api_key:
            raise RuntimeError("set ICR_API_BASE and ICR_API_KEY (OpenAI-compatible endpoint) before calling the API")
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai (or pip install -e '.[api]')") from e
        self._client = OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key, timeout=self.settings.timeout_s)
        self.retries = retries
        self.usage = Usage()
        self._sem = threading.Semaphore(self.settings.max_concurrency)
        self.no_temperature: set[str] = set()  # models whose endpoint rejects an explicit temperature (learned at run time)

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.0, max_tokens: int = 2048, **kw) -> str:
        model = model or self.settings.generator_model
        delay = 2.0
        last: Exception | None = None
        for _attempt in range(self.retries):
            try:
                with self._sem:
                    r = self._client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kw
                    )
                u = getattr(r, "usage", None)
                self.usage.add(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - any transport error is retried
                last = e
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
        raise RuntimeError(f"API call failed after {self.retries} attempts: {last}")

    def complete(
        self,
        system: str | None,
        user: str,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout_s: float | None = None,
        extra_body: dict | None = None,
        **kw,
    ) -> dict:
        """One chat completion returned as a RECORD (never raises): content, finish_reason, usage, latency, attempts, error.

        Retries with exponential back-off on 408/409/429/5xx and transport errors; when the endpoint rejects an explicit
        temperature for a model (HTTP 400 mentioning 'temperature') the parameter is dropped for that model and the call
        is repeated at once; `temperature_applied` records what was sent. Used by the exercise generator / verifier /
        auditor so that every API record in the data files has the same shape.
        """
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        t0 = time.time()
        attempts = 0
        err: str | None = None
        client = self._client.with_options(timeout=timeout_s) if timeout_s else self._client
        for k in range(self.retries):
            attempts += 1
            params: dict = {"max_completion_tokens": max_tokens}
            if extra_body:
                params["extra_body"] = extra_body
            params.update(kw)
            if temperature is not None and model not in self.no_temperature:
                params["temperature"] = temperature
            try:
                with self._sem:
                    r = client.chat.completions.create(model=model, messages=msgs, **params)
                ch = r.choices[0]
                u = r.usage.model_dump() if getattr(r, "usage", None) else {}
                det = u.get("completion_tokens_details") or {}
                self.usage.add(u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0)
                return {
                    "content": ch.message.content or "",
                    "finish_reason": ch.finish_reason,
                    "usage": {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "reasoning_tokens": det.get("reasoning_tokens")},
                    "latency": time.time() - t0,
                    "temperature_applied": params.get("temperature"),
                    "attempts": attempts,
                    "served_model": getattr(r, "model", None),
                    "error": None,
                }
            except Exception as e:  # noqa: BLE001 - every failure is recorded, retried or returned
                msg = str(e)
                err = f"{type(e).__name__}: {msg[:200]}"
                status = getattr(e, "status_code", None)
                if status == 400 and "temperature" in msg:
                    self.no_temperature.add(model)
                    continue
                retry = status in (408, 409, 429) or (status is not None and status >= 500) or status is None
                if not retry or k == self.retries - 1:
                    break
                time.sleep(min(64, 2 ** (k + 1)) + random.random())
        return {"content": "", "finish_reason": None, "usage": {}, "latency": time.time() - t0, "temperature_applied": None, "attempts": attempts, "served_model": None, "error": err}
