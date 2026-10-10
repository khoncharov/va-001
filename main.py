"""
Точка входа умной колонки (прототип на Gemma 4 E2B + Ollama).

Запуск:
    python main.py                    # push-to-talk (Enter -> фраза -> ответ)
    python main.py --wake             # wake word («Алиса …»)
    python main.py --calibrate        # калибровка порога тишины микрофона
    python main.py --devices          # список аудиоустройств
    python main.py --test-llm         # текстовый smoke-тест LLM без микрофона
    python main.py --test-audio       # отправить test.wav в STT (проверка формата аудио)
    python main.py --no-tts           # отключить озвучку (только текст)
    python main.py -v                 # подробные логи
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import CONFIG


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)-8s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_devices() -> None:
    from audio_io import list_devices

    list_devices()


def cmd_calibrate() -> None:
    """
    Показывает уровень громкости с микрофона в реальном времени.
    Помогает выставить VadConfig.silence_rms: замерь «шум комнаты» и «речь»,
    поставь порог между ними (обычно ~3x от шума).
    """
    import time

    from audio_io import Microphone

    mic = Microphone()
    peak = {"v": 0.0}

    def on_level(rms: float) -> None:
        peak["v"] = max(peak["v"], rms)
        bar = "█" * min(60, int(rms * 600))
        print(f"\r  rms={rms:.4f} {bar:<60}", end="", flush=True)

    mic.level_callback = on_level
    mic.start()
    print("Калибровка. Считай 10 секунд ШУМ (молчи), потом 10 секунд ГОВОРИ.")
    try:
        for phase, secs in (("ШУМ (молчи)", 10), ("РЕЧЬ (говори/пой)", 10)):
            peak["v"] = 0.0
            print(f"\n>>> Фаза: {phase} — {secs} сек")
            t0 = time.monotonic()
            while time.monotonic() - t0 < secs:
                time.sleep(0.1)
            print(f"\n    пик за фазу: rms={peak['v']:.4f}")
            globals().setdefault("_phase_peaks", []).append((phase, peak["v"]))
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()

    peaks = globals().get("_phase_peaks", [])
    if len(peaks) == 2:
        noise, speech = peaks[0][1], peaks[1][1]
        suggested = max(noise * 2.5, (noise + speech) / 3)
        print("\n=== Рекомендация ===")
        print(f"  шум: {noise:.4f} | речь: {speech:.4f}")
        print(f"  попробуй SMARTSPEAKER_SILENCE_RMS={suggested:.4f}")
        print("  (Windows PowerShell): $env:SMARTSPEAKER_SILENCE_RMS='%.4f'" % suggested)
        print("  (bash/zsh): export SMARTSPEAKER_SILENCE_RMS=%.4f" % suggested)


def cmd_test_llm() -> None:
    """Без аудио: проверяем, что модель вообще отвечает."""
    from ollama_client import OllamaClient

    oc = OllamaClient()
    if not oc.check():
        sys.exit(1)
    q = "Привет! Как дела? Ответь одним предложением."
    print(f">>> {q}")
    answer = oc.chat([], q)
    print(f"<<< {answer}")


def cmd_test_audio(path: str) -> None:
    """Проверяем STT на готовом WAV — изолированно от микрофона и VAD."""
    from ollama_client import OllamaClient

    p = Path(path)
    if not p.exists():
        sys.exit(f"Файл не найден: {path}")
    oc = OllamaClient()
    if not oc.check():
        sys.exit(1)
    import time

    t0 = time.monotonic()
    text = oc.transcribe(p.read_bytes())
    print(f"STT за {time.monotonic()-t0:.1f}s: {text!r}")


def cmd_run(args: argparse.Namespace) -> None:
    from speaker import SmartSpeaker

    if not args.tts:
        CONFIG.tts.enabled = False

    sp = SmartSpeaker(use_wake=args.wake)
    sp.run()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Умная колонка (прототип)")
    ap.add_argument("--wake", action="store_true", help="режим wake word вместо push-to-talk")
    ap.add_argument("--no-tts", dest="tts", action="store_false", default=True,
                    help="не озвучивать ответы, только текст")
    ap.add_argument("--calibrate", action="store_true", help="калибровка порога тишины")
    ap.add_argument("--devices", action="store_true", help="список микрофонов/динамиков")
    ap.add_argument("--test-llm", action="store_true", help="smoke-тест LLM (текст)")
    ap.add_argument("--test-audio", metavar="WAV", help="smoke-тест STT на WAV-файле")
    ap.add_argument("-v", "--verbose", action="store_true", help="подробные логи")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.devices:
        cmd_devices()
    elif args.calibrate:
        cmd_calibrate()
    elif args.test_llm:
        cmd_test_llm()
    elif args.test_audio:
        cmd_test_audio(args.test_audio)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
