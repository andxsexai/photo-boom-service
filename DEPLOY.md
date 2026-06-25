# Деплой Photo Boom — публичная ссылка 24/7

Сейчас сервис работает **только на вашем Mac** (`localhost:8000`).  
Чтобы получить ссылку вида `https://photo-boom-....` — нужен **один** деплой в облако.

---

## Вариант 1 — Render (рекомендуется, 1 клик)

1. Откройте: **[Deploy to Render](https://render.com/deploy?repo=https://github.com/andxsexai/photo-boom-service)**
2. Войдите через GitHub → подтвердите создание сервиса
3. Render соберёт Docker из репозитория
4. Получите URL: `https://photo-boom-xxxx.onrender.com`

В `render.yaml` уже прописаны переменные:
- `USE_HF_FALLBACK=true` — face restore через HuggingFace (без вашего ComfyUI)
- `USE_LOCAL_MODELS=false`

> На бесплатном Render сервис «засыпает» после 15 мин без трафика — первый запрос ~30 сек.

---

## Вариант 2 — Fly.io (быстрее, нужна карта)

```bash
cd photo-boom-service
fly auth login
fly launch --copy-config --yes   # fly.toml уже в репо
fly deploy
```

URL: **https://photo-boom-andx.fly.dev**

> Fly запросил карту на аккаунте `andxsex@gmail.com` — без неё деплoy не создаётся.

---

## Вариант 3 — Hugging Face Spaces (бесплатно, для ML)

1. https://huggingface.co/new-space → SDK: **Docker**
2. Имя: `photo-boom` · Visibility: Public
3. Подключите GitHub repo `andxsexai/photo-boom-service` или:

```bash
git remote add space https://huggingface.co/spaces/andxsexai/photo-boom
git push space main
```

URL: **https://huggingface.co/spaces/andxsexai/photo-boom**

---

## Локально vs Облако

| | Локально (Mac) | Облако |
|---|---|---|
| URL | localhost:8000 | https://.... |
| CodeFormer | ✅ ComfyUI + MPS | ⚠️ HuggingFace fallback |
| Работает без Mac | ❌ | ✅ |
| Демо-примеры | ✅ | ✅ (в Docker образе) |

---

## После деплоя

Проверка:
```bash
curl https://ВАШ-URL/health
# {"status":"ok","mode":"cloud",...}
```

Добавьте в `.env` на сервере (если нужен Telegram-бот):
```
TELEGRAM_BOT_TOKEN=...
PUBLIC_URL=https://ваш-url
```
