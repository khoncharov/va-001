"""
Клиент Ollama: STT (аудио->текст), LLM (текст->текст), TTS (текст->аудио).

Важные детали протокола, которые стоит проверить в обзоре:

1. STT. Ollama принимает аудио ТОЛЬКО через OpenAI-совместимый эндпоинт
   POST /v1/chat/completions, где сообщение содержит content-массив с блоком
   {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}.
   Нативный /api/generate и /api/chat аудио НЕ принимают. Это ограничение Ollama.

2. LLM. Обычный /api/chat (не стриминг — для простоты прототипа; задержка
   первого токена на CPU всё равно велика, см. метрики).

3. TTS. Экспериментально: /api/generate с полем "audio": {"format": "wav"},
   ответ приходит в поле "audio" как base64. Поддерживается не всеми моделями
   (например pocket-tts). Если TTS недоступен — колонка печатает ответ в консоль.

Все три шага логируются по времени в logs/metrics.jsonl — из них и складывается
главное ограничение голосового интерфейса: total latency от конца фразы до ответа.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from config import CONFIG

log = logging.getLogger("ollama")


@dataclass
class Timings:
    """Замеры одного полного цикла «услышала -> ответила»."""
    stt_s: float | None = None
    llm_s: float | None = None
    tts_s: float | None = None
    total_s: float | None = None

    def as_dict(self) -> dict:
        return {k: round(v, 3) if v is not None else None for k, v in self.__dict__.items()}


class OllamaClient:
    def __init__(self) -> None:
        self._host = CONFIG.llm.ollama_host.rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                CONFIG.llm.generate_timeout, connect=CONFIG.llm.connect_timeout
            )
        )

    # ------------------------------------------------------------------ health
    def check(self) -> bool:
        """Pinged сервер и печатаем список моделей — полезно при первом запуске."""
        try:
            r = self._client.get(f"{self._host}/api/tags", timeout=5.0)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            log.info("Ollama доступен. Модели: %s", ", ".join(names) or "— пусто —")
            want = CONFIG.llm.model
            if not any(n == want or n.split(":")[0] == want.split(":")[0] for n in names):
                log.warning(
                    "Модель '%s' не найдена среди загруженных. "
                    "Выполни: ollama pull %s (или поправь SMARTSPEAKER_MODEL)",
                    want, want,
                )
            return True
        except Exception as e:
            log.error("Ollama недоступен по адресу %s: %s", self._host, e)
            return False

    # --------------------------------------------------------------------- STT
    def transcribe(self, wav_bytes: bytes) -> str:
        cfg = CONFIG.stt
        b64 = base64.b64encode(wav_bytes).decode()
        payload = {
            "model": cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg.prompt_hint},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        },
                    ],
                }
            ],
            "stream": False,
        }
        t0 = time.monotonic()
        r = self._client.post(
            f"{self._host}{cfg.endpoint}", json=payload, timeout=cfg.timeout
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        log.debug("STT за %.1fс: %r", time.monotonic() - t0, text)
        return text

    # --------------------------------------------------------------------- LLM
    def chat(self, history: list[dict], user_text: str) -> str:
        cfg = CONFIG.llm
        messages = [{"role": "system", "content": cfg.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": cfg.num_ctx, "temperature": cfg.temperature},
        }
        t0 = time.monotonic()
        r = self._client.post(f"{self._host}/api/chat", json=payload, timeout=cfg.generate_timeout)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "").strip()
        dur = time.monotonic() - t0
        # нативная статистика Ollama: load_time (загрузка модели!), eval_count/duration
        stats = {
            k: data.get(k)
            for k in ("load_duration", "prompt_eval_count", "eval_count", "eval_duration")
        }
        log.debug("LLM за %.1fс: %r | %s", dur, text[:80], stats)
        return text

    # --------------------------------------------------------------------- TTS
    def speak(self, text: str) -> bytes | None:
        """Возвращает WAV-байты или None, если TTS недоступен/выключен."""
        cfg = CONFIG.tts
        if not cfg.enabled:
            return None
        payload = {
            "model": cfg.model,
            "prompt": text,
            "stream": False,
            "audio": {"format": cfg.format},
        }
        try:
            t0 = time.monotonic()
            r = self._client.post(
                f"{self._host}/api/generate", json=payload, timeout=cfg.timeout
            )
            r.raise_for_status()
            data = r.json()
            audio_b64 = data.get("audio")
            if not audio_b64:
                log.warning("Модель TTS '%s' вернула ответ без аудио-поля.", cfg.model)
                return None
            wav = base64.b64decode(audio_b64)
            log.debug("TTS за %.1fс, %d байт", time.monotonic() - t0, len(wav))
            return wav
        except httpx.HTTPStatusError as e:
            log.warning(
                "TTS недоступен (HTTP %s): %s. Ответ будет только текстом.",
                e.response.status_code, e.response.text[:200],
            )
            return None
        except Exception as e:
            log.warning("TTS ошибка: %s. Ответ будет только текстом.", e)
            return None

    # ------------------------------------------------------- полный цикл + метрики
    def process_cycle(self, history: list[dict], wav_bytes: bytes) -> tuple[str, str, bytes | None, Timings]:
        """STT -> LLM -> TTS за один вызов. Возвращает (вопрос, ответ, wav, тайминги)."""
        t = Timings()
        wall0 = time.monotonic()

        t0 = time.monotonic(); question = self.transcribe(wav_bytes); t.stt_s = time.monotonic() - t0
        t0 = time.monotonic(); answer = self.chat(history, question); t.llm_s = time.monotonic() - t0
        t0 = time.monotonic(); wav = self.speak(answer); t.tts_s = time.monotonic() - t0

        t.total_s = time.monotonic() - wall0
        return question, answer, wav, t
