"""
Логика пробуждения по wake word без отдельной offline-модели.

Компромиссное решение для прототипа: микрофон работает постоянно, кольцевой
буфер хранит последние N секунд; когда VAD видит фразу — транскрибируем её
через ту же мультимодальную модель и проверяем, есть ли wake-фраза в начале.
Если нет — считаем это случайным шумом/разговором и идём дальше слушать.

Ограничения (важно для обзора):
  * каждое ложное срабатывание VAD стоит вызова STT -> расход CPU/батареи;
  * задержка реакции выше, чем у специализированных offline-детекторов
    (Picovoice Porcupine / openWakeWord);
  * качество зависит от того, насколько надёжно модель распознаёт само слово.

Для production лучше заменить на openWakeWord (бесплатный) или Porcupine.
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np

from config import CONFIG

log = logging.getLogger("wake")

_NON_WORD = re.compile(r"[^\w\s\-]", re.UNICODE)


def normalize(text: str) -> str:
    """Для «сравнения» транскрипта с wake-фразой: только буквы/цифры, нижний регистр."""
    t = _NON_WORD.sub(" ", text.lower())
    return " ".join(t.split())


def contains_wake(transcript: str, phrase: str | None = None, prefix_chars: int | None = None) -> bool:
    phrase = normalize(phrase or CONFIG.wake.wake_phrase)
    if not phrase:
        return True
    norm = normalize(transcript)
    window = norm[: (prefix_chars or CONFIG.wake.check_prefix_chars)]
    # допускаем варианты написания: "алиса" / "алиса," внутри окна
    return phrase in window


class WakeGate:
    """
    Обёртка над кольцевым буфером + STT для проверки наличия wake-слова.

    Метод listen_once(recorder_pcm) вызывается после того, как PhraseRecorder
    уже собрал фразу; gate решает, реагировать или игнорировать.
    """

    def __init__(self, transcribe_fn) -> None:
        self._transcribe = transcribe_fn
        self.last_check_s: float | None = None

    def accept(self, pcm: np.ndarray, wav_bytes: bytes) -> tuple[bool, str]:
        t0 = time.monotonic()
        try:
            transcript = self._transcribe(wav_bytes)
        except Exception as e:
            log.error("Wake-проверка: STT упал (%s). Игнорирую фразу.", e)
            return False, ""
        self.last_check_s = time.monotonic() - t0
        ok = contains_wake(transcript)
        log.info(
            "Wake-проверка %.2fс -> %s | транскрипт: %r",
            self.last_check_s, "СРАБОТАЛО" if ok else "нет", transcript[:120],
        )
        return ok, transcript
