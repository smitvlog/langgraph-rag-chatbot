from dataclasses import replace

from fastapi.testclient import TestClient

import app.main as main
from app.main import app, docchat

client = TestClient(app)


def new_session() -> str:
    return client.post("/api/sessions").json()["session_id"]


def test_upload_lists_the_file_with_its_chunk_count():
    session = new_session()
    res = client.post(
        f"/api/sessions/{session}/files",
        files={"files": ("policy.txt", b"Remote work is allowed three days a week.", "text/plain")},
    )
    assert res.status_code == 200
    assert res.json() == {"files": {"policy.txt": 1}}
    assert client.get(f"/api/sessions/{session}/files").json() == {"files": {"policy.txt": 1}}


def test_unsupported_upload_is_rejected():
    res = client.post(
        f"/api/sessions/{new_session()}/files",
        files={"files": ("photo.png", b"\x89PNG", "image/png")},
    )
    assert res.status_code == 415


def test_chat_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(docchat, "ask", lambda session, message: {"answer": "ok", "sources": [], "route": "chat"})
    res = client.post(f"/api/sessions/{new_session()}/chat", json={"message": "hi"})
    assert res.json()["answer"] == "ok"


def test_llm_failure_is_reported_as_502(monkeypatch):
    def boom(session, message):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(docchat, "ask", boom)
    res = client.post(f"/api/sessions/{new_session()}/chat", json={"message": "hi"})
    assert res.status_code == 502


def test_unknown_session_format_is_404():
    assert client.get("/api/sessions/../../etc/files").status_code == 404
    assert client.get("/api/sessions/not-a-session/files").status_code == 404


def test_web_ui_is_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "DocChat" in res.text


def test_upload_over_the_chunk_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, max_chunks_per_session=1))
    session = new_session()
    text = b"Remote work is allowed three days a week. " * 40
    res = client.post(f"/api/sessions/{session}/files", files={"files": ("long.txt", text, "text/plain")})
    assert res.status_code == 413
    assert client.get(f"/api/sessions/{session}/files").json() == {"files": {}}


def test_sample_handbook_can_be_loaded():
    session = new_session()
    res = client.post(f"/api/sessions/{session}/sample")
    assert res.status_code == 200
    assert res.json() == {"files": {"northwind-employee-handbook.pdf": 7}}


def test_info_reports_whether_tracing_is_on():
    assert client.get("/api/info").json() == {"tracing": main.settings.tracing}
