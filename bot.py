"""Telegram bot for Photo Boom (Aiogram 3.x)."""

from __future__ import annotations

import asyncio
import io
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)

from config import settings
from core.enhancer import PhotoEnhancer
from core.filters import STYLE_META, Style, all_styles
from core.session import SessionLimitError, SessionManager

logger = logging.getLogger(__name__)

STYLE_BUTTONS = {
    Style.UPGRADE: "01 Upgrade",
    Style.QUIET_LUXURY: "02 Quiet Luxury",
    Style.CINEMATIC_TWILIGHT: "03 Cinematic Twilight",
    Style.BW_NOIR: "04 B&W Noir",
    Style.MAGAZINE_COVER: "05 Magazine Cover",
}


def style_keyboard(session_id: str) -> InlineKeyboardMarkup:
    rows = []
    for style in all_styles():
        rows.append(
            [
                InlineKeyboardButton(
                    text=STYLE_BUTTONS[style],
                    callback_data=f"process:{session_id}:{style.value}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="⚡ Все 5 стилей",
                callback_data=f"process_all:{session_id}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def main() -> None:
    if not settings.telegram_bot_token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    sessions = SessionManager(limit=settings.session_limit)
    enhancer = PhotoEnhancer(
        use_local_models=settings.use_local_models,
        use_hf_fallback=settings.use_hf_fallback,
        hf_space=settings.hf_space,
        codeformer_fidelity=settings.codeformer_fidelity,
    )

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        session = sessions.create()
        webapp_url = f"https://your-domain.com/?session={session.id}"
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📱 Open WebApp", web_app=WebAppInfo(url=webapp_url))]
            ]
        )
        await message.answer(
            "🎬 *PHOTO BOOM* — ANDX NETWORK\n\n"
            f"Отправьте фото — получите до *{settings.session_limit}* премиум-обработок.\n\n"
            "Стили:\n"
            "1️⃣ Upgrade\n"
            "2️⃣ Quiet Luxury\n"
            "3️⃣ Cinematic Twilight\n"
            "4️⃣ B&W Noir\n"
            "5️⃣ Magazine Cover",
            parse_mode="Markdown",
            reply_markup=kb,
        )

    @dp.message(F.photo)
    async def handle_photo(message: Message) -> None:
        session = sessions.create()
        await message.answer(
            f"📸 Фото получено. Выберите стиль ({session.remaining}/{session.limit} доступно):",
            reply_markup=style_keyboard(session.id),
        )
        # Store photo file_id keyed by session for callback handlers
        if not hasattr(handle_photo, "_photos"):
            handle_photo._photos = {}
        handle_photo._photos[session.id] = message.photo[-1].file_id

    @dp.callback_query(F.data.startswith("process:"))
    async def cb_process(callback) -> None:
        _, session_id, style_value = callback.data.split(":", 2)
        style = Style(style_value)
        session = sessions.get(session_id)
        if not session or not session.can_process(style):
            await callback.answer("Лимит исчерпан или стиль уже использован", show_alert=True)
            return

        photos = getattr(handle_photo, "_photos", {})
        file_id = photos.get(session_id)
        if not file_id:
            await callback.answer("Сначала отправьте фото", show_alert=True)
            return

        await callback.answer("Обрабатываю…")
        status = await callback.message.answer("⏳ GFPGAN → Real-ESRGAN → LUT…")

        file = await callback.bot.get_file(file_id)
        buf = io.BytesIO()
        await callback.bot.download_file(file.file_path, buf)
        buf.seek(0)

        from PIL import Image

        image = Image.open(buf).convert("RGB")
        try:
            session.consume(style)
        except SessionLimitError:
            await status.edit_text("❌ Лимит сессии исчерпан")
            return

        result = enhancer.process_single(image, style)
        out = io.BytesIO()
        result.image.save(out, format="JPEG", quality=95)
        out.seek(0)

        await status.delete()
        await callback.message.answer_photo(
            BufferedInputFile(out.read(), filename=f"{style.value}.jpg"),
            caption=(
                f"✅ *{STYLE_META[style]['name']}*\n"
                f"Модель: `{result.face_model}` | Осталось: {session.remaining}/{session.limit}"
            ),
            parse_mode="Markdown",
            reply_markup=style_keyboard(session_id) if session.remaining > 0 else None,
        )

    @dp.callback_query(F.data.startswith("process_all:"))
    async def cb_process_all(callback) -> None:
        session_id = callback.data.split(":", 1)[1]
        session = sessions.get(session_id)
        if not session or session.remaining <= 0:
            await callback.answer("Лимит исчерпан", show_alert=True)
            return

        photos = getattr(handle_photo, "_photos", {})
        file_id = photos.get(session_id)
        if not file_id:
            await callback.answer("Сначала отправьте фото", show_alert=True)
            return

        await callback.answer("Обрабатываю все 5…")
        status = await callback.message.answer("⏳ Полный пайплайн…")

        file = await callback.bot.get_file(file_id)
        buf = io.BytesIO()
        await callback.bot.download_file(file.file_path, buf)
        buf.seek(0)

        from PIL import Image

        image = Image.open(buf).convert("RGB")
        pending = [s for s in all_styles() if session.can_process(s)]
        pipeline = enhancer.process_all(image, styles=pending)

        await status.delete()
        for variant in pipeline.variants:
            try:
                session.consume(variant.style)
            except SessionLimitError:
                break
            out = io.BytesIO()
            variant.image.save(out, format="JPEG", quality=95)
            out.seek(0)
            await callback.message.answer_photo(
                BufferedInputFile(out.read(), filename=f"{variant.style.value}.jpg"),
                caption=f"✅ {STYLE_META[variant.style]['name']}",
            )

        await callback.message.answer(
            f"Готово! Использовано: {session.used}/{session.limit}",
        )

    logger.info("Photo Boom bot starting…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
