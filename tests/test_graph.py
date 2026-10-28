"""Graph wiring tests. The LLM is replaced by a scripted fake, so no API key is needed;
embeddings are real (the model is cached locally after the first run)."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import app.graph as graph_module
from app.graph import NOT_FOUND, DocChat
from app.store import SessionStore

LEAVE = Document(
    "Employees get 24 days of paid annual leave per calendar year.",
    metadata={"source": "handbook.pdf", "page": 2},
)


def scripted(monkeypatch, *replies):
    fake = FakeListChatModel(responses=list(replies))
    monkeypatch.setattr(graph_module, "get_llm", lambda: fake)


def test_without_documents_it_just_chats(monkeypatch):
    scripted(monkeypatch, "Hi! Upload a document to get started.")
    result = DocChat(SessionStore()).ask("s1", "hello")
    assert result["route"] == "chat"
    assert result["sources"] == []


def test_document_question_is_answered_with_sources(monkeypatch):
    store = SessionStore()
    store.add("s1", "handbook.pdf", [LEAVE])
    scripted(monkeypatch, "docs", "You get 24 days of paid leave [1].")

    result = DocChat(store).ask("s1", "How many days of annual leave do I get?")

    assert result["answer"] == "You get 24 days of paid leave [1]."
    assert result["sources"][0]["source"] == "handbook.pdf"
    assert result["sources"][0]["page"] == 2


def test_full_width_citations_are_normalised(monkeypatch):
    store = SessionStore()
    store.add("s1", "handbook.pdf", [LEAVE])
    scripted(monkeypatch, "docs", "You get 24 days of paid leave【1】.")

    result = DocChat(store).ask("s1", "How many days of annual leave do I get?")

    assert result["answer"] == "You get 24 days of paid leave[1]."


def test_unrelated_question_is_not_answered_from_documents(monkeypatch):
    store = SessionStore()
    store.add("s1", "handbook.pdf", [LEAVE])
    scripted(monkeypatch, "docs")

    result = DocChat(store).ask("s1", "What is the boiling point of tungsten in kelvin?")

    assert result["answer"] == NOT_FOUND
    assert result["sources"] == []


def test_follow_up_is_rewritten_using_history(monkeypatch):
    store = SessionStore()
    store.add("s1", "handbook.pdf", [LEAVE])
    docchat = DocChat(store)
    scripted(monkeypatch, "docs", "24 days [1].")
    docchat.ask("s1", "How much annual leave do I get?")

    # rewrite → route → answer; the rewritten question is what gets searched.
    scripted(monkeypatch, "How much annual leave do part-time employees get?", "docs", "Pro rata [1].")
    result = docchat.ask("s1", "and part-timers?")

    assert result["answer"] == "Pro rata [1]."
    assert result["sources"][0]["page"] == 2


def test_sessions_do_not_share_documents(monkeypatch):
    store = SessionStore()
    store.add("s1", "handbook.pdf", [LEAVE])
    scripted(monkeypatch, "Please upload a document first.")

    result = DocChat(store).ask("s2", "How many days of annual leave do I get?")

    assert result["route"] == "chat"


def test_idle_sessions_are_forgotten(monkeypatch):
    store = SessionStore()
    chat = DocChat(store, ttl_seconds=60)
    now = [0.0]
    monkeypatch.setattr(graph_module, "monotonic", lambda: now[0])
    chat.touch("s1")
    store.add("s1", "handbook.pdf", [LEAVE])

    now[0] = 61
    chat.touch("s2")

    assert store.files("s1") == {}


def test_least_recently_used_session_is_forgotten_over_the_cap():
    store = SessionStore()
    chat = DocChat(store, max_sessions=2)
    for session in ("s1", "s2"):
        chat.touch(session)
        store.add(session, "handbook.pdf", [LEAVE])

    chat.touch("s1")
    chat.touch("s3")

    assert store.files("s1") == {"handbook.pdf": 1}
    assert store.files("s2") == {}


def test_summary_reads_across_the_document_instead_of_searching(monkeypatch):
    store = SessionStore()
    pages = [Document(f"Section {n} of the handbook.", metadata={"source": "handbook.pdf", "page": n}) for n in range(1, 4)]
    store.add("s1", "handbook.pdf", pages)
    scripted(monkeypatch, "summary", "The handbook has three sections [1][2][3].")

    result = DocChat(store).ask("s1", "Summarise this document")

    assert result["route"] == "summary"
    assert [s["page"] for s in result["sources"]] == [1, 2, 3]


def test_overview_spreads_chunks_over_the_document():
    store = SessionStore()
    chunks = [Document(f"chunk {n}", metadata={"source": "a.txt", "page": n}) for n in range(10)]
    store.add("s1", "a.txt", chunks)

    assert [d.metadata["page"] for d in store.overview("s1", 5)] == [0, 2, 4, 6, 8]
    assert len(store.overview("s1", 20)) == 10


def test_answer_that_cites_nothing_shows_no_sources(monkeypatch):
    store = SessionStore()
    store.add("s1", "handbook.pdf", [LEAVE])
    scripted(monkeypatch, "docs", "The excerpts don't say how much annual leave interns get.")

    result = DocChat(store).ask("s1", "How much annual leave do interns get?")

    assert result["sources"] == []
