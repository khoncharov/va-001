"""
VAD (Voice Activity Detection) на энергии сигнала + кольцевой буфер.

Два режима:
  * record_phrase() — ждать речь, записать фразу до паузы (wake-режим / push-to-talk);
  * RingBuffer      — постоянно хранить последние N секунд аудио (для wake word).

Это самый простой детектор: он не отличает речь от шума пылесоса или телевизора.
Для прототипа этого достаточно, но это одно из явных ограничений — см. README.
"""
from __future__ import annotations

import collections
import queue
import time

import numpy as np

from audio_io import AudioBlock
from config import CONFIG


class RingBuffer:
    """Хранит последние `seconds` аудио в блоках."""

    def __init__(self, seconds: float, block_seconds: float) -> None:
        maxlen = max(1, int(seconds / block_seconds))
        self._buf: collections.deque[AudioBlock] = collections.deque(maxlen=maxlen)
        self._block_seconds = block_seconds

    def push(self, block: AudioBlock) -> None:
        self._buf.append(block)

    def get_all(self) -> np.ndarray:
        if not self._buf:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([b.samples for b in self._buf])

    def clear(self) -> None:
        self._buf.clear()


class PhraseRecorder:
    """
    Собирает блоки из очереди, пока RMS выше порога; завершает по тишине.

    Логика состояний:
      WAITING  — ждём начала речи (RMS > silence_rms)
      SPEAKING — пишем; держим preroll (кусочек до первого слова)
      DONE     — silence_tail_s секунд тишины или max_phrase_s -> хватит
    """

    WAITING = "waiting"
    SPEAKING = "speaking"
    DONE = "done"
    TIMEOUT = "timeout"   # так и не дождались речи
    CANCELLED = "cancelled"

    def __init__(self, q: queue.Queue[AudioBlock], time_fn=time.monotonic) -> None:
        self._q = q
        self._cfg = CONFIG.vad
        self._audio_cfg = CONFIG.audio
        self._time = time_fn           # можно подставить фейковое время в тестах
        self._block_s = self._audio_cfg.blocksize / self._audio_cfg.sample_rate
        self._preroll_n = max(1, int(self._cfg.preroll_s / self._block_s))
        self._preroll: collections.deque[AudioBlock] = collections.deque(maxlen=self._preroll_n)
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._last_voice_time: float | None = None
        self.state = self.WAITING
        self.cancel_event_flag = False

    # -- один шаг обработки блока -------------------------------------------------
    def feed(self, block: AudioBlock) -> None:
        now = self._time()
        voiced = block.rms >= self._cfg.silence_rms

        if self.state == self.WAITING:
            self._preroll.append(block)
            if voiced:
                # в preroll держим не более preroll_n блоков ДО начала речи;
                # сам voiced-блок добавляем один раз — как первый блок фразы.
                # (deque к этому моменту уже содержит block, поэтому -1)
                keep_n = max(0, min(len(self._preroll) - 1, self._preroll_n))
                self.state = self.SPEAKING
                for b in list(self._preroll)[-keep_n:]:
                    self._add(b.samples)
                self._preroll.clear()
                self._add(block.samples)
                self._last_voice_time = now
        elif self.state == self.SPEAKING:
            # хвост тишины НЕ включаем в запись фразы (модели хватает, а метрика
            # длительности остаётся честной); молчаливый шум не раздувает STT
            if voiced:
                self._add(block.samples)
                self._last_voice_time = now
            else:
                assert self._last_voice_time is not None
                if now - self._last_voice_time >= self._cfg.silence_tail_s:
                    self.state = self.DONE
            if self._total_samples / self._audio_cfg.sample_rate >= self._cfg.max_phrase_s:
                self.state = self.DONE

    def _add(self, samples: np.ndarray) -> None:
        self._chunks.append(samples)
        self._total_samples += samples.size

    # -- основной цикл -------------------------------------------------------------
    def run(self, wait_timeout_s: float | None = None) -> np.ndarray | None:
        """
        Блокирующе читает очередь. Возвращает float32 PCM фразы или None
        (таймаут ожидания речи / слишком коротко / отменено).
        """
        start = self._time()
        while True:
            if self.cancel_event_flag:
                self.state = self.CANCELLED
                return None
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                if (
                    wait_timeout_s is not None
                    and self.state == self.WAITING
                    and self._time() - start > wait_timeout_s
                ):
                    self.state = self.TIMEOUT
                    return None
                continue

            self.feed(block)

            if self.state in (self.DONE, self.TIMEOUT, self.CANCELLED):
                break

        pcm = (
            np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        )
        dur = pcm.size / self._audio_cfg.sample_rate
        if dur < self._cfg.min_phrase_s:
            self.state = self.TIMEOUT
            return None
        return pcm
