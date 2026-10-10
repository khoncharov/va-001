"""
Тесты ядра колонки БЕЗ железа и без Ollama.

Проверяют:
  * упаковка/распаковка WAV;
  * VAD-логика на фейковом времени (preroll, хвост тишины, отброс коротких фраз);
  * wake-word матчинг транскриптов;
  * формат HTTP-payload'ов для STT/LLM/TTS Ollama (моки).

Запуск:  python -m pytest test_core.py   (или просто python test_core.py)
"""
from __future__ import annotations

import base64
import json
import queue

import numpy as np

from audio_io import AudioBlock, pcm_to_wav_bytes, wav_bytes_to_pcm
from config import CONFIG
from vad import PhraseRecorder, RingBuffer


# ----------------------------------------------------------------------------- WAV
def test_wav_roundtrip():
    pcm = (np.sin(np.linspace(0, 100, 8000)) * 0.4).astype(np.float32)
    wav = pcm_to_wav_bytes(pcm, 16000)
    back, sr = wav_bytes_to_pcm(wav)
    assert sr == 16000
    assert abs(len(back) - len(pcm)) <= 1
    assert float(np.max(np.abs(back - pcm))) < 0.001


# ------------------------------------------------------------------ фейковое время
class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _recorder_with(blocks_rms):
    """Создаёт PhraseRecorder с очередью блоков и фейковыми временными метками.

    Часы перематываются НА МОМЕНТ ВЫДАЧИ блока (как при реальном стриминге),
    поэтому тесты проверяют и логику таймеров, и содержимое буферов.
    """
    clock = FakeClock()

    class TimedQueue(queue.Queue):
        def __init__(self, items):
            super().__init__()
            self._items = list(items)   # [(t, AudioBlock)]
            for it in self._items:
                super().put(it)

        def get(self, timeout=None):
            item = super().get(timeout=timeout)
            t, block = item
            clock.t = t                 # «сейчас» = момент выпуска этого блока
            return block

    # раскладываем блоки по времени их реальной «записи»
    stamps = []
    t = clock.t
    block_dt = CONFIG.audio.blocksize / CONFIG.audio.sample_rate
    for rms in blocks_rms:
        samples = np.full(CONFIG.audio.blocksize, rms, dtype=np.float32)
        stamps.append((t, AudioBlock(samples, rms)))
        t += block_dt

    tq = TimedQueue(stamps)
    rec = PhraseRecorder(tq, time_fn=lambda: clock.t)  # ленивое чтение часов
    return rec, tq


# --------------------------------------------------------------------------- VAD
def test_vad_records_phrase_with_preroll_and_tail():
    silence_n = 5
    speech_n = 10                       # 1 секунда речи
    tail_needed = int(CONFIG.vad.silence_tail_s / 0.1) + 1
    rec, _ = _recorder_with(
        [0.001] * silence_n + [0.2] * speech_n + [0.001] * tail_needed
    )
    pcm = rec.run(wait_timeout_s=999)
    assert pcm is not None and rec.state == PhraseRecorder.DONE
    dur = pcm.size / CONFIG.audio.sample_rate
    # preroll(0.3с) + речь(1.0с) + часть хвоста тишины
    assert CONFIG.vad.preroll_s + 1.0 <= dur <= CONFIG.vad.preroll_s + 1.0 + CONFIG.vad.silence_tail_s + 0.15


def test_vad_rejects_short_blip():
    tail_needed = int(CONFIG.vad.silence_tail_s / 0.1) + 1
    # 2 блока речи => 0.2с < min_phrase_s (0.4с) -> фразу отбрасываем как шум
    rec, _ = _recorder_with([0.2] * 2 + [0.001] * tail_needed)
    assert rec.run(wait_timeout_s=999) is None


def test_vad_max_phrase_cap():
    n = int(CONFIG.vad.max_phrase_s / 0.1) + 5
    rec, _ = _recorder_with([0.2] * n)
    pcm = rec.run(wait_timeout_s=999)
    assert pcm is not None
    assert pcm.size / CONFIG.audio.sample_rate <= CONFIG.vad.max_phrase_s + 0.1


def test_ring_buffer_capacity():
    rb = RingBuffer(2.0, 0.1)
    for _ in range(50):
        rb.push(AudioBlock(np.ones(1600, dtype=np.float32), 1.0))
    assert rb.get_all().size == 2 * CONFIG.audio.sample_rate


# -------------------------------------------------------------------- wake word
def test_wake_matching():
    from wakeword import contains_wake, normalize

    assert normalize("  Алиса, включи Свет!! ") == "алиса включи свет"
    assert contains_wake("Алиса, какая погода?")
    assert contains_wake("алиса! покажи время")
    # «алиса» не в начале окна — это обычный разговор, будить колонку нельзя
    assert not contains_wake("привет подруге про алису вчера вечером было интересно")
    assert not contains_wake("включи музыку")


# ------------------------------------------------------- протокол Ollama (моки)
class FakeResp:
    def __init__(self, data):
        self._d = data
        self.status_code = 200
        self.text = json.dumps(data)

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, url, **kw):
        return FakeResp({"models": [{"name": CONFIG.llm.model}]})

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if url.endswith("/v1/chat/completions"):
            return FakeResp({"choices": [{"message": {"content": "Включи свет"}}]})
        if url.endswith("/api/chat"):
            return FakeResp(
                {"message": {"content": "Уже включаю."}, "eval_count": 9}
            )
        return FakeResp({"audio": base64.b64encode(b"RIFFfake").decode()})


def test_ollama_protocol_shapes():
    from ollama_client import OllamaClient

    oc = OllamaClient()
    oc._client = FakeClient()
    assert oc.check() is True

    question, answer, wav, timings = oc.process_cycle([], b"WAVDATA")
    assert question == "Включи свет"
    assert answer == "Уже включаю."
    assert wav == b"RIFFfake"
    assert timings.total_s is not None

    stt_url, stt_payload = oc._client.calls[0]
    assert stt_url.endswith("/v1/chat/completions"), "STT обязан идти через OpenAI-эндпоинт"
    blocks = stt_payload["messages"][0]["content"]
    audio_block = blocks[1]
    assert audio_block["type"] == "input_audio"
    assert audio_block["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio_block["input_audio"]["data"]) == b"WAVDATA"

    chat_url, chat_payload = oc._client.calls[1]
    assert chat_url.endswith("/api/chat")
    assert chat_payload["options"]["num_ctx"] == CONFIG.llm.num_ctx
    assert chat_payload["messages"][0]["role"] == "system"

    tts_url, tts_payload = oc._client.calls[2]
    assert tts_url.endswith("/api/generate")
    assert tts_payload["audio"] == {"format": CONFIG.tts.format}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} тестов пройдено")
    raise SystemExit(1 if failed else 0)
