"""REST endpoints for document upload, retrieval, annotation, and export."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from app.services.document_store import get_document_store
from app.services.exporter import export_docx, export_pdf
from config.settings import get_settings

router = APIRouter(prefix="/documents", tags=["documents"])

_MAX_SIZE = 5 * 1024 * 1024  # 5 MB


class HighlightRequest(BaseModel):
    sentence_idx: int
    color: str = "yellow"


class SnippetRequest(BaseModel):
    term: str
    explanation: str
    sentence_idx: int = -1
    word_idx: int = -1


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "document.md"
    if not filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")

    content = await file.read()
    if len(content) > _MAX_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 5 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc = get_document_store().save_document(filename, content)
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "title": doc.title,
        "word_count": doc.word_count,
        "sentence_count": doc.sentence_count,
        "uploaded_at": doc.uploaded_at,
    }


@router.get("/")
async def list_documents() -> list[dict]:
    return get_document_store().list_documents()


@router.get("/{doc_id}")
async def get_document(doc_id: str) -> dict:
    doc = get_document_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    annotations = get_document_store().load_annotations(doc_id)
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "title": doc.title,
        "raw_markdown": doc.raw_markdown,
        "sentences": doc.sentences,
        "word_count": doc.word_count,
        "sentence_count": doc.sentence_count,
        "uploaded_at": doc.uploaded_at,
        "annotations": annotations,
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    removed = get_document_store().delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@router.get("/{doc_id}/annotations")
async def get_annotations(doc_id: str) -> dict:
    doc = get_document_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return get_document_store().load_annotations(doc_id)


@router.post("/{doc_id}/annotations/highlight")
async def add_highlight(doc_id: str, body: HighlightRequest) -> dict:
    doc = get_document_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    get_document_store().save_highlight(doc_id, body.sentence_idx, body.color)
    return {"saved": True, "sentence_idx": body.sentence_idx}


@router.post("/{doc_id}/annotations/snippet")
async def add_snippet(doc_id: str, body: SnippetRequest) -> dict:
    doc = get_document_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    snippet = get_document_store().save_snippet(
        doc_id=doc_id,
        term=body.term,
        explanation=body.explanation,
        sentence_idx=body.sentence_idx,
        word_idx=body.word_idx,
    )
    return snippet


@router.get("/{doc_id}/export/{fmt}")
async def export_document(
    doc_id: str,
    fmt: Literal["pdf", "docx"],
    background_tasks: BackgroundTasks,
) -> FileResponse:
    doc = get_document_store().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    annotations = get_document_store().load_annotations(doc_id)
    settings = get_settings()
    settings.exports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w-]", "_", doc.title[:30]) if doc.title else "document"
    filename = f"{safe_title}_{ts}.{fmt}"
    output_path = settings.exports_dir / filename

    loop = asyncio.get_event_loop()
    try:
        if fmt == "pdf":
            await loop.run_in_executor(None, export_pdf, doc, annotations, output_path)
            media_type = "application/pdf"
        else:
            await loop.run_in_executor(None, export_docx, doc, annotations, output_path)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    except Exception as exc:
        logger.warning("event=export_failed doc_id={} fmt={} error={}", doc_id, fmt, exc)
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    background_tasks.add_task(lambda: output_path.unlink(missing_ok=True))
    return FileResponse(output_path, media_type=media_type, filename=filename)
