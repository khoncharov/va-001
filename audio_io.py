"""
Низкоуровневая работа со звуком: запись с микрофона, воспроизведение, WAV.

Используется sounddevice (PortAudio). Если библиотека не установилась —
смотри README, раздел «Установка».
"""
from __future__ import annotations

import io
import logging
import queue
import threading
import wave
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
except OSError as e:  # PortAudio не найден в системе
    sd = None
    _SD_IMPORT_ERROR = e
else:
    _SD_IMPORT_ERROR = None

from config import CONFIG

log = logging.getLogger("audio")


def ensure_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "Не удалось загрузить PortAudio (sounddevice). "
            f"Исходная ошибка: {_SD_IMPORT_ERROR}. "
            "См. README: на Windows установи 'pip install sounddevice', "
            "на Linux нужен пакет libportaudio2."
        )


# ---------------------------------------------------------------------------
# WAV: упаковка/распаковка int16 PCM
# ---------------------------------------------------------------------------
def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1..1] моно -> байты WAV (PCM 16-bit)."""
    clipped = np.clip(pcm, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())
    return buf.getvalue()


def wav_bytes_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    """Байты WAV -> (float32 массив, sample_rate). Нужен для проигрывания ответа TTS."""
    with wave.open(io.BytesIO(data), "rb") as r:
        sr = r.getframerate()
        n = r.getnframes()
        raw = r.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[r.getsampwidth()]
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if r.getnchannels() > 1:  # на случай стерео из TTS — усредняем до моно
        arr = arr.reshape(-1, r.getnchannels()).mean(axis=1)
    scale = np.iinfo(dtype).max if dtype != np.uint8 else 127.0
    return arr / scale, sr


def save_wav(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    with open(path, "wb") as f:
        f.write(pcm_to_wav_bytes(pcm, sample_rate))


# ---------------------------------------------------------------------------
# Микрофон: отдельный поток + очередь блоков
# ---------------------------------------------------------------------------
@dataclass
class AudioBlock:
    samples: np.ndarray          # float32mono
    rms: float                   # root mean square блока


class Microphone:
    """
    Кольцевой источник аудио. Каждый слушатель получает СВОЮ очередь блоков —
    это позволяет одновременно работать VAD-записи и фоновому кольцевому буферу wake-режима.
    """

    def __init__(self) -> None:
        ensure_sounddevice()
        self._cfg = CONFIG.audio
        self._streams: list[queue.Queue[AudioBlock]] = []
        self._lock = threading.Lock()
        self._stream = None
        self.level_callback = None  # fn(rms) — для индикатора громкости/калибровки

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("sounddevice status: %s", status)
        mono = indata[:, 0].astype(np.float32)
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        block = AudioBlock(mono.copy(), rms)
        if self.level_callback:
            try:
                self.level_callback(rms)
            except Exception:
                pass
        with self._lock:
            for q in self._streams:
                # не растёт бесконечно при зависании потребителя
                if q.qsize() > 200:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put(block)

    def subscribe(self) -> queue.Queue[AudioBlock]:
        q: queue.Queue[AudioBlock] = queue.Queue()
        with self._lock:
            self._streams.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[AudioBlock]) -> None:
        with self._lock:
            if q in self._streams:
                self._streams.remove(q)

    def start(self) -> None:
        cfg = self._cfg
        log.info(
            "Открываю микрофон: rate=%d, device=%s", cfg.sample_rate, cfg.input_device
        )
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=cfg.blocksize,
            device=cfg.input_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def list_devices() -> None:
    ensure_sounddevice()
    print(sd.query_devices())


# ---------------------------------------------------------------------------
# Динамик
# ---------------------------------------------------------------------------
def play_pcm(pcm: np.ndarray, sample_rate: int) -> None:
    ensure_sounddevice()
    sd.play(pcm, samplerate=sample_rate, blocking=True)


def play_wav_bytes(data: bytes) -> None:
    pcm, sr = wav_bytes_to_pcm(data)
    play_pcm(pcm, sr)
