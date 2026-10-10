"""
Конфигурация умной колонки (прототип).

Все настройки в одном месте, чтобы их было удобно переключать во время обзора
возможностей и ограничений модели Gemma 4 E2B под CPU (Intel Core Ultra 5 225H).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Модель / Ollama
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    # Имя модели в Ollama. Поменяй на реальное после `ollama pull`,
    # если оно отличается (например "gemma4:e2b", "gemma-e2b:latest", ...).
    model: str = os.getenv("SMARTSPEAKER_MODEL", "gemma4:e2b")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Параметры генерации. Для CPU-инференса держим контекст небольшим —
    # это основной источник тормозов на ноутбуке без дискретной видеокарты.
    num_ctx: int = int(os.getenv("SMARTSPEAKER_NUM_CTX", "2048"))
    temperature: float = float(os.getenv("SMARTSPEAKER_TEMPERATURE", "0.7"))

    # Сколько ходов диалога помнить (1 ход = вопрос + ответ).
    max_history_turns: int = 6

    # Таймауты HTTP к Ollama (сек). Первая загрузка модели на CPU может быть долгой.
    connect_timeout: float = 5.0
    generate_timeout: float = 120.0

    # Системный промпт: короткий, разговорный, запрет на markdown/списки —
    # ответ всё равно озвучивается через TTS.
    system_prompt: str = (
        "Ты — голосовой ассистент умной колонки. Отвечай по-русски, коротко "
        "(1–3 предложения), разговорным тоном, без markdown, списков и эмодзи. "
        "Если не уверен — честно скажи об этом."
    )


# ---------------------------------------------------------------------------
# Аудио: запись / воспроизведение
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    sample_rate: int = 16_000          # 16 кГц — стандарт для распознавания речи
    channels: int = 1                  # моно (встроенный микрофон ноутбука)
    blocksize: int = 1600              # 100 мс на блок при 16 кГц

    # Индексы устройств; None = устройство по умолчанию ОС.
    input_device: int | None = None
    output_device: int | None = None

    # Воспроизведение всегда в 16 кГц (формат WAV от Ollama TTS).
    play_sample_rate: int = 16_000


# ---------------------------------------------------------------------------
# VAD: детектор тишины для остановки записи
# ---------------------------------------------------------------------------
@dataclass
class VadConfig:
    # RMS ниже этого порога считается тишиной. Порог зависит от микрофона
    # и уровня шума в комнате — обязательно проверь через --calibrate!
    silence_rms: float = float(os.getenv("SMARTSPEAKER_SILENCE_RMS", "0.010"))

    silence_tail_s: float = 1.2        # сколько секунд тишины = конец фразы
    max_phrase_s: float = 15.0         # жёсткий лимит длины фразы
    min_phrase_s: float = 0.4          # короче — считать шумом, не отправлять
    preroll_s: float = 0.3             # буфер «до» начала речи, чтобы не съесть первую фонему


# ---------------------------------------------------------------------------
# Пробуждение ("wake word")
# ---------------------------------------------------------------------------
@dataclass
class WakeConfig:
    enabled: bool = os.getenv("SMARTSPEAKER_WAKE", "1") == "1"
    # Ожидаемая транскрипция wake-фразы (ниже регистр, без знаков).
    wake_phrase: str = "алиса"
    # Фраза должна встретиться в начале транскрипта (первые N символов).
    check_prefix_chars: int = 25
    # Сколько последних секунд аудио держать в кольцевом буфере ожидания.
    ring_seconds: float = 5.0


# ---------------------------------------------------------------------------
# STT / TTS
# ---------------------------------------------------------------------------
@dataclass
class STTConfig:
    # Ollama принимает аудио только через OpenAI-совместимый эндпоинт
    # /v1/chat/completions (поле content как base64 input_audio).
    endpoint: str = "/v1/chat/completions"
    # Если модель НЕ поддерживает аудио нативно, можно указать отдельную
    # speech-to-text модель, например " whisper-cpp".
    model: str = os.getenv("SMARTSPEAKER_STT_MODEL") or os.getenv(
        "SMARTSPEAKER_MODEL", "gemma4:e2b"
    )
    prompt_hint: str = "Распиши услышанное в тексте максимально точно."
    timeout: float = 60.0


@dataclass
class TTSConfig:
    enabled: bool = os.getenv("SMARTSPEAKER_TTS", "1") == "1"
    # Ollama экспериментально поддерживает TTS: /api/generate с "audio": {"format": "wav"}
    # Работает пока лишь с некоторыми моделями (pocket-tts и т.п.).
    model: str = os.getenv("SMARTSPEAKER_TTS_MODEL", "pocket-tts")
    format: str = "wav"
    timeout: float = 60.0


# ---------------------------------------------------------------------------
# Прочее
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    log_dir: str = "logs"              # сюда складываем WAV-записи для разбора
    metrics_path: str = "logs/metrics.jsonl"


CONFIG = AppConfig()
