# PHOTO BOOM — ANDX NETWORK

Микросервис автоматической премиум-обработки фото: **5 стилей** на одну сессию.

| № | Стиль | Описание |
|---|-------|----------|
| 1 | **Upgrade** | Чистая ретушь + детализация без изменения цвета |
| 2 | **Quiet Luxury** | Бежево-графитовый LUT |
| 3 | **Cinematic Twilight** | Глубокие тени, голливудский контраст |
| 4 | **B&W Noir** | Монохром, зерно, глянцевый портрет |
| 5 | **Magazine Cover** | Quiet Luxury + минималистичный оверлей |

## Архитектура

```
Исходное фото
   │
   ▼
[CodeFormer fidelity 0.5–0.7] ──► fallback GFPGAN при артефактах
   │
   ▼
[Real-ESRGAN x4plus] ──► Lanczos fallback на CPU
   │
   ├──► Upgrade
   ├──► Quiet Luxury (+ 3D LUT)
   ├──► Cinematic Twilight (+ 3D LUT + vignette)
   ├──► B&W Noir (+ grain)
   └──► Magazine Cover (Quiet Luxury + overlay)
```

## Структура

```
photo-boom-service/
├── models/download_weights.py   # GFPGAN / CodeFormer / Real-ESRGAN
├── core/
│   ├── filters.py               # 5 стилей обработки
│   ├── enhancer.py              # нейросети + HF fallback
│   ├── fallback.py              # детектор артефактов
│   └── session.py               # лимит 5 обработок
├── static/index.html            # WebApp UI
├── app.py                       # FastAPI
├── bot.py                       # Telegram bot (Aiogram)
├── Dockerfile
└── requirements.txt
```

## Публичная ссылка (облако 24/7)

**Сейчас:** сервис работает только локально — `http://localhost:8000`  
**Чтобы работал без Mac:** задеплойте один раз → см. **[DEPLOY.md](DEPLOY.md)**

| Платформа | URL после деплоя | Сложность |
|-----------|------------------|-----------|
| [Render](https://render.com/deploy?repo=https://github.com/andxsexai/photo-boom-service) | `photo-boom.onrender.com` | 1 клик |
| Fly.io | `photo-boom-andx.fly.dev` | CLI + карта |
| HuggingFace Spaces | `huggingface.co/spaces/andxsexai/photo-boom` | бесплатно |

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/andxsexai/photo-boom-service)

## Быстрый старт (локально)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Запуск API + WebApp
uvicorn app:app --reload --port 8000

# Telegram bot (нужен TELEGRAM_BOT_TOKEN)
python bot.py
```

Откройте http://localhost:8000

## Docker

```bash
docker build -t photo-boom .
docker run -p 8000:8000 photo-boom
```

## Загрузка весов (GPU-сервер)

```bash
python models/download_weights.py all
# USE_LOCAL_MODELS=true в .env
```

## MVP без GPU

По умолчанию `USE_HF_FALLBACK=true` — face restoration через HuggingFace Space:

`avans06/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN`

Стили и LUT применяются локально через OpenCV/PIL.

## API

| Method | Endpoint | Описание |
|--------|----------|----------|
| POST | `/session` | Создать сессию (лимит 5) |
| GET | `/session/{id}` | Статус сессии |
| GET | `/styles` | Список стилей |
| POST | `/process` | Обработать 1 стиль (`session_id`, `style`, `file`) |
| POST | `/preview/all` | **Предпросмотр всех 5 стилей** (без лимита) |
| POST | `/preview/{style}` | Предпросмотр одного стиля |
| POST | `/preview/compare` | Оригинал + все превью для UI-слайдера |
| POST | `/process/all` | Все доступные стили за раз |

| GET | `/download/{session_id}/{style}` | Скачать результат |

## Предпросмотр

После загрузки фото WebApp автоматически вызывает `/preview/all` — показывает пример всех 5 стилей **без списания лимита**. Слайдер «Оригинал / Предпросмотр» помогает сравнить до финальной обработки.

Backend жёстко ограничивает **5 обработок** на сессию. Каждый стиль можно применить только один раз. Лимит не доверяется фронтенду.

## Выгрузка на GitHub

```bash
git init
git add .
git commit -m "feat: initial commit for Photo Boom service with 5 premium styles"
git remote add origin https://github.com/andxsexai/photo-boom-service.git
git branch -M main
git push -u origin main
```

## Лицензия

ANDX NETWORK © 2025
