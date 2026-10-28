"""HTTP API and the static web UI.

Run with:  uvicorn app.main:app --reload
"""

import logging
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .graph import DocChat
from .ingest import UnsupportedFile, load_file, split
from .store import SessionStore

log = logging.getLogger("docchat")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "northwind-employee-handbook.pdf"
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")

app = FastAPI(title="DocChat", description="Chat with your documents (RAG + LangGraph)")
docchat = DocChat(SessionStore())


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def check_session(session_id: str) -> str:
    if not SESSION_ID.match(session_id):
        raise HTTPException(404, "Unknown session")
    return session_id


@app.post("/api/sessions")
def create_session() -> dict:
    return {"session_id": uuid.uuid4().hex}


@app.get("/api/sessions/{session_id}/files")
def list_files(session_id: str) -> dict:
    return {"files": docchat.store.files(check_session(session_id))}


# Plain `def` endpoints run in FastAPI's thread pool, so embedding and LLM calls
# don't block the event loop.
@app.post("/api/sessions/{session_id}/files")
def upload_files(session_id: str, files: list[UploadFile] = File(...)) -> dict:
    check_session(session_id)
    docchat.touch(session_id)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    for upload in files:
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"{upload.filename} is larger than {settings.max_upload_mb} MB")
        add_document(session_id, upload.filename, data)
    return {"files": docchat.store.files(session_id)}


@app.post("/api/sessions/{session_id}/sample")
def load_sample(session_id: str) -> dict:
    """Loads the bundled sample handbook, so visitors can try DocChat without a file of their own."""
    check_session(session_id)
    docchat.touch(session_id)
    add_document(session_id, SAMPLE.name, SAMPLE.read_bytes())
    return {"files": docchat.store.files(session_id)}


def add_document(session_id: str, filename: str, data: bytes) -> None:
    try:
        pages = load_file(filename, data)
    except UnsupportedFile as err:
        raise HTTPException(415, str(err)) from err
    if not pages:
        raise HTTPException(422, f"{filename} has no readable text (is it a scanned PDF?)")
    chunks = split(pages)
    if sum(docchat.store.files(session_id).values()) + len(chunks) > settings.max_chunks_per_session:
        raise HTTPException(413, f"{filename} is too long for this chat. Try a shorter document.")
    docchat.store.add(session_id, filename, chunks)


@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: str, request: ChatRequest) -> dict:
    check_session(session_id)
    try:
        return docchat.ask(session_id, request.message)
    except Exception as err:  # the LLM provider can fail (rate limits, bad key, network)
        log.exception("chat failed")
        raise HTTPException(502, "The language model could not answer right now. Please try again.") from err


@app.delete("/api/sessions/{session_id}")
def reset_session(session_id: str) -> dict:
    docchat.forget(check_session(session_id))
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
