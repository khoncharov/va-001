# Как запустить умную колонку на локальном компьютере

Прототип: ноутбук ASUS S14 (Intel Core Ultra 5 225H, без дискретной GPU),
модель **gemma4:e2b** через **Ollama**, Python.

---

## Шаг 0. Требования

- Python **3.10+** (проверка: `python --version`)
- Установленный и запущенный **Ollama**: https://ollama.com/download
- Рабочий микрофон и динамики в ОС

## Шаг 1. Виртуальное окружение и зависимости

В папке проекта (там, где лежит `main.py`):

**Windows (PowerShell):**
```powershell
cd C:\путь\к\проекту
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
cd /путь/к/проекту
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Если PowerShell ругается на политику выполнения скриптов:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, затем активировать снова.

Для разработчиков: дополнительно можно поставить `pytest` для юнит-тестов
(`pip install pytest`, запуск: `pytest test_core.py`).

## Шаг 2. Ollama + модель

```bash
ollama serve            # обычно стартует автоматически как сервис
ollama pull gemma4:e2b  # если имя модели отличается — посмотри ollama list
```

Проверка, что сервер жив:

```bash
curl http://localhost:11434/api/tags     # Windows: Invoke-RestMethod http://localhost:11434/api/tags
```

Если имя модели в `ollama list` отличается от `gemma4:e2b`, не правь код —
перевыпусти переменную окружения:

```bash
export SMARTSPEAKER_MODEL="имя:тег"          # Linux/macOS
$env:SMARTSPEAKER_MODEL = "имя:тег"          # Windows PowerShell
```

## Шаг 3. Проверка аудиоустройств и калибровка VAD

```bash
python main.py --devices      # посмотреть индексы микрофона/динамиков
python main.py --calibrate    # 10 сек молчи, 10 сек говори — смотри уровни RMS
```

По результатам калибровки подстрой порог тишины (по умолчанию 0.010):

```bash
export SMARTSPEAKER_SILENCE_RMS=0.02   # или значение, подобное из calibrate
```

## Шаг 4. Smoke-тесты перед полноценным запуском

```bash
python main.py --test-llm     # текстовый запрос к LLM (микрофон не нужен)
python main.py --test-audio   # отправить тестовый WAV в STT (проверка формата аудио)
```

## Шаг 5. Запуск

```bash
python main.py                # push-to-talk: Enter -> говоритcя фраза -> ответ
python main.py --wake         # режим wake word («Алиса …») — VAD слушает постоянно
python main.py --no-tts       # ответы только текстом, без озвучки
python main.py -v             # подробные логи (latency, размеры чанков и т.п.)
```

Комбинации флагов работают вместе, например:

```bash
python main.py --wake --no-tts -v
```

Остановка: `Ctrl+C`.

## Частые проблемы

| Симптом | Причина / решение |
|---|---|
| `ConnectionError` / таймаут к Ollama | `ollama serve` не запущен; проверь `OLLAMA_HOST` |
| Модель не найдена (404) | Имя не совпадает с `ollama list` → `SMARTSPEAKER_MODEL` |
| Не пишет микрофон / пустой WAV | Нет доступа к микрофону в настройках ОС; другой индекс устройства → `--devices`, затем `AudioConfig.input_device` в `config.py` |
| Фраза «не заканчивается» или обрывается | Неправильный порог VAD → `--calibrate`, затем `SMARTSPEAKER_SILENCE_RMS` |
| Ответы генерируются очень медленно | На CPU это ожидаемо. Уменьши `SMARTSPEAKER_NUM_CTX` (например, до 1024) или поставь quantization полегче |
| `PortAudioError` при установке sounddevice | Windows: обычно работает из коробки; Linux: `sudo apt install libportaudio2` |

## Переменные окружения (все настройки без правки кода)

- `SMARTSPEAKER_MODEL` — имя LLM-модели в Ollama (по умолчанию `gemma4:e2b`)
- `OLLAMA_HOST` — адрес сервера (по умолчанию `http://localhost:11434`)
- `SMARTSPEAKER_NUM_CTX` — размер контекста (по умолчанию `2048`)
- `SMARTSPEAKER_TEMPERATURE` — температура генерации (по умолчанию `0.7`)
- `SMARTSPEAKER_SILENCE_RMS` — порог тишины VAD (по умолчанию `0.010`)
- `SMARTSPEAKER_WAKE` — wake word: `1` вкл / `0` выкл (по умолчанию `1`)
- `SMARTSPEAKER_STT_MODEL` — отдельная модель STT (если не задана, используется основная)
- `SMARTSPEAKER_TTS` — озвучка ответов: `1` вкл / `0` выкл (по умолчанию `1`)
- `SMARTSPEAKER_TTS_MODEL` — модель TTS в Ollama (по умолчанию `pocket-tts`)

Примечание: `--test-audio` принимает путь к WAV-файлу, например
`python main.py --test-audio record.wav`.
