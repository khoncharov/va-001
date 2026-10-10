"""
Состояния колонки и главный цикл (конечный автомат).

    IDLE ──(Enter / wake-фраза)──► LISTENING ──(пауза по VAD)──► THINKING
      ▲                                │                             │
      └─────────(ответ озвучен)────────┴────◄ SPEAKING ◄─────────────┘

Два режима:
  * push-to-talk (по умолчанию): жмёшь Enter, говоришь — проще отлаживать STT/латентность;
  * wake word (--wake): слушает всегда, реагирует только если фраза начинается с wake-слова.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

import audio_io
from audio_io import Microphone, play_wav_bytes, save_wav
from config import CONFIG
from ollama_client import OllamaClient, Timings
from vad import PhraseRecorder
from wakeword import WakeGate

log = logging.getLogger("speaker")


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


def _banner(state: State) -> None:
    marks = {
        State.IDLE: "⏸  ожидание",
        State.LISTENING: "🎙  СЛУШАЮ…",
        State.THINKING: "🧠  ДУМАЮ…",
        State.SPEAKING: "🔊  ОТВЕЧАЮ…",
    }
    print(f"\n[{marks[state]}]", flush=True)


class SmartSpeaker:
    def __init__(self, use_wake: bool = False) -> None:
        self.state = State.IDLE
        self.use_wake = use_wake and CONFIG.wake.enabled
        self.history: list[dict] = []          # [{role, content}, ...]
        self._stop = threading.Event()

        self.mic = Microphone()
        self.ollama = OllamaClient()
        self.wake_gate = WakeGate(self.ollama.transcribe)

        self.log_dir = Path(CONFIG.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_path = Path(CONFIG.metrics_path)
        self.metrics_path.parent.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _set(self, s: State) -> None:
        self.state = s
        _banner(s)

    def _record_phrase(self, wait_timeout_s: float | None = None) -> np.ndarray | None:
        q = self.mic.subscribe()
        try:
            rec = PhraseRecorder(q)
            return rec.run(wait_timeout_s=wait_timeout_s)
        finally:
            self.mic.unsubscribe(q)

    def _append_history(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        max_msgs = CONFIG.llm.max_history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def _write_metrics(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ core cycle
    def handle_audio(self, pcm: np.ndarray, pretranscribed: str | None = None) -> None:
        """Один полный цикл: WAV -> STT -> LLM -> TTS/печать."""
        wav = audio_io.pcm_to_wav_bytes(pcm, CONFIG.audio.sample_rate)

        # сохраняем запись — пригодится для разбора косяков распознавания
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = self.log_dir / f"rec_{stamp}.wav"
        save_wav(str(rec_path), pcm, CONFIG.audio.sample_rate)

        self._set(State.THINKING)
        timings = Timings()
        t0 = time.monotonic()
        try:
            if pretranscribed is not None:
                question = pretranscribed
            else:
                question = self.ollama.transcribe(wav)
            timings.stt_s = time.monotonic() - t0

            t0 = time.monotonic()
            answer = self.ollama.chat(self.history, question)
            timings.llm_s = time.monotonic() - t0

            t0 = time.monotonic()
            tts_wav = self.ollama.speak(answer)
            timings.tts_s = time.monotonic() - t0
        except Exception as e:
            log.exception("Ошибка в цикле обработки")
            print(f"   ⚠ Ошибка: {e}\n   (Ollama запущен? модель загружена?)")
            self._set(State.IDLE)
            return

        print(f"   🗣  Ты: {question}")
        print(f"   🤖 Ответ: {answer}")

        if tts_wav:
            self._set(State.SPEAKING)
            try:
                play_wav_bytes(tts_wav)
            except Exception as e:
                log.warning("Не удалось воспроизвести TTS: %s", e)

        timings.total_s = (timings.stt_s or 0) + (timings.llm_s or 0) + (timings.tts_s or 0)
        print(
            "   ⏱ STT %.1fs | LLM %.1fs | TTS %s | итого ~%.1fs"
            % (
                timings.stt_s or 0,
                timings.llm_s or 0,
                f"{timings.tts_s:.1f}s" if timings.tts_s is not None else "нет",
                timings.total_s or 0,
            )
        )
        self._append_history(question, answer)
        self._write_metrics(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "recording": str(rec_path),
                "audio_len_s": round(pcm.size / CONFIG.audio.sample_rate, 2),
                "question": question,
                "answer": answer[:300],
                **timings.as_dict(),
            }
        )
        self._set(State.IDLE)

    # ------------------------------------------------------------------ modes
    def run_ppt(self) -> None:
        """Push-to-talk: Enter -> запись фразы -> обработка."""
        print("Режим push-to-talk. Enter — говорить, Ctrl+C — выход.")
        while not self._stop.is_set():
            try:
                cmd = input("\n⏸  [Enter] — слушать  |  'clear' — сброс диалога  |  Ctrl+C — выход: ")
            except EOFError:
                break
            if cmd.strip().lower() in ("clear", "сброс"):
                self.history.clear()
                print("История очищена.")
                continue

            self._set(State.LISTENING)
            pcm = self._record_phrase(wait_timeout_s=8.0)
            if pcm is None:
                print("   … тишина, не услышала речь.")
                self._set(State.IDLE)
                continue
            self.handle_audio(pcm)

    def run_wake(self) -> None:
        """Wake-word: слушаем всегда, реагируем на фразу-пробуждение в начале реплики."""
        wake = CONFIG.wake.wake_phrase
        print(f"Режим wake word: скажи «{wake} …». Ctrl+C — выход.")
        while not self._stop.is_set():
            self._set(State.IDLE)
            self._set(State.LISTENING)
            pcm = self._record_phrase()
            if pcm is None:
                continue
            wav = audio_io.pcm_to_wav_bytes(pcm, CONFIG.audio.sample_rate)
            ok, transcript = self.wake_gate.accept(pcm, wav)
            if not ok:
                log.info("Игнорирую (нет wake-слова): %r", transcript[:80])
                continue
            # вырезаем wake-слово из начала и отправляем остальное в диалог
            rest = transcript
            norm = transcript.lower().replace(wake, "", 1)
            rest = norm.strip(" ,.!?")
            if rest:
                # уже есть транскрипт — не зовём STT второй раз
                self.handle_audio(pcm, pretranscribed=rest)
            else:
                # просто разбудили — ждём команду без wake-слова
                self._set(State.LISTENING)
                pcm2 = self._record_phrase(wait_timeout_s=6.0)
                if pcm2 is not None:
                    self.handle_audio(pcm2)

    # ------------------------------------------------------------------ lifecycle
    def run(self) -> None:
        if not self.ollama.check():
            raise SystemExit(
                "Ollama недоступен. Установи и запусти: https://ollama.com\n"
                "затем: ollama pull " + CONFIG.llm.model
            )
        self.mic.start()
        mode = self.run_wake if self.use_wake else self.run_ppt
        try:
            mode()
        except KeyboardInterrupt:
            print("\nПока! 🖐")
        finally:
            self._stop.set()
            self.mic.stop()
