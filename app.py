"""Photo Boom — FastAPI backend with session limits and 5 premium styles."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import base64

from config import settings
from core.enhancer import PhotoEnhancer
from core.filters import STYLE_META, Style, all_styles, image_to_bytes
from core.image_utils import load_image_normalized
from core.session import SessionLimitError, SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Photo Boom",
    description="ANDX NETWORK — 5 premium photo processing styles",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = SessionManager(limit=settings.session_limit)
enhancer = PhotoEnhancer(
    use_local_models=settings.use_local_models,
    use_hf_fallback=settings.use_hf_fallback,
    hf_space=settings.hf_space,
    codeformer_fidelity=settings.codeformer_fidelity,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class SessionResponse(BaseModel):
    session_id: str
    limit: int
    remaining: int


class StyleInfo(BaseModel):
    id: str
    num: int
    name: str
    description: str


class ProcessResponse(BaseModel):
    session_id: str
    style: str
    remaining: int
    face_model: str
    upscaled: bool
    download_url: str


class ProcessAllResponse(BaseModel):
    session_id: str
    remaining: int
    face_model: str
    upscaled: bool
    variants: list[ProcessResponse]


class PreviewItem(BaseModel):
    style: str
    name: str
    num: int
    thumbnail: str  # data:image/jpeg;base64,...


class PreviewAllResponse(BaseModel):
    previews: list[PreviewItem]
    note: str = "Предпросмотр — пример стиля. Финальная обработка в полном качестве."


def _load_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, WebP)")
    data = file.file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 20 MB)")
    try:
        return load_image_normalized(data)
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc


def _to_data_uri(image: Image.Image, quality: int = 80) -> str:
    b64 = base64.b64encode(image_to_bytes(image, quality=quality)).decode()
    return f"data:image/jpeg;base64,{b64}"


def _save_output(session_id: str, style: Style, image: Image.Image) -> Path:
    out_dir = settings.output_dir / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{style.value}.jpg"
    image.save(path, format="JPEG", quality=95, optimize=True)
    return path


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"service": "Photo Boom", "brand": "ANDX NETWORK", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "brand": "ANDX NETWORK"}


@app.get("/styles", response_model=list[StyleInfo])
async def list_styles():
    return [
        StyleInfo(
            id=s.value,
            num=STYLE_META[s]["num"],
            name=STYLE_META[s]["name"],
            description=STYLE_META[s]["description"],
        )
        for s in all_styles()
    ]


@app.post("/session", response_model=SessionResponse)
async def create_session():
    session = sessions.create()
    return SessionResponse(
        session_id=session.id,
        limit=session.limit,
        remaining=session.remaining,
    )


@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")
    return SessionResponse(
        session_id=session.id,
        limit=session.limit,
        remaining=session.remaining,
    )


@app.post("/process", response_model=ProcessResponse)
async def process_single(
    session_id: str = Form(...),
    style: Style = Form(...),
    file: UploadFile = File(...),
):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")
    if not session.can_process(style):
        raise HTTPException(
            429,
            detail={
                "error": "session_limit",
                "message": "Processing limit reached or style already used",
                "remaining": session.remaining,
            },
        )

    image = _load_image(file)
    try:
        session.consume(style)
    except SessionLimitError as exc:
        raise HTTPException(429, str(exc)) from exc

    result = enhancer.process_single(image, style)
    _save_output(session_id, style, result.image)

    return ProcessResponse(
        session_id=session_id,
        style=style.value,
        remaining=session.remaining,
        face_model=result.face_model,
        upscaled=result.upscaled,
        download_url=f"/download/{session_id}/{style.value}",
    )


@app.post("/process/all", response_model=ProcessAllResponse)
async def process_all(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")

    pending = [s for s in all_styles() if session.can_process(s)]
    if not pending:
        raise HTTPException(
            429,
            detail={"error": "session_limit", "remaining": session.remaining},
        )

    image = _load_image(file)
    pipeline = enhancer.process_all(image, styles=pending)

    variants: list[ProcessResponse] = []
    for variant in pipeline.variants:
        try:
            session.consume(variant.style)
        except SessionLimitError:
            break
        _save_output(session_id, variant.style, variant.image)
        variants.append(
            ProcessResponse(
                session_id=session_id,
                style=variant.style.value,
                remaining=session.remaining,
                face_model=variant.face_model,
                upscaled=variant.upscaled,
                download_url=f"/download/{session_id}/{variant.style.value}",
            )
        )

    return ProcessAllResponse(
        session_id=session_id,
        remaining=session.remaining,
        face_model=pipeline.face_model_used,
        upscaled=pipeline.upscaled,
        variants=variants,
    )


@app.get("/download/{session_id}/{style}")
async def download(session_id: str, style: str):
    path = settings.output_dir / session_id / f"{style}.jpg"
    if not path.exists():
        raise HTTPException(404, "Processed image not found")
    return FileResponse(path, media_type="image/jpeg", filename=f"photo-boom-{style}.jpg")


@app.post("/preview/all", response_model=PreviewAllResponse)
async def preview_all_styles(file: UploadFile = File(...)):
    """Fast preview of all 5 styles — does NOT consume session limit."""
    image = _load_image(file)
    previews = enhancer.preview_all(image)
    return PreviewAllResponse(
        previews=[
            PreviewItem(
                style=p.style.value,
                name=STYLE_META[p.style]["name"],
                num=STYLE_META[p.style]["num"],
                thumbnail=_to_data_uri(p.image),
            )
            for p in previews
        ]
    )


@app.post("/preview/{style}")
async def preview_style(style: Style, file: UploadFile = File(...)):
    """Single style preview — no session limit."""
    image = _load_image(file)
    result = enhancer.preview_single(image, style)
    return Response(content=image_to_bytes(result.image, quality=85), media_type="image/jpeg")


@app.post("/preview/compare")
async def preview_compare(file: UploadFile = File(...)):
    """Return original + active style side-by-side data for UI slider."""
    image = _load_image(file)
    thumb = enhancer.preview_all(image)
    return {
        "original": _to_data_uri(image),
        "previews": {p.style.value: _to_data_uri(p.image) for p in thumb},
    }
