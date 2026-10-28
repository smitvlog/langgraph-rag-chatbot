"""The LangGraph workflow.

    START → rewrite → route ─┬─ chat ───────────────────────────→ END
                             └─ retrieve ─┬─ answer (with sources) → END
                                          └─ not_found ──────────→ END

`route` picks chat, docs (search for the chunks closest to the question) or summary
(chunks spread across the whole document, for "summarise this" or "tell me about Smit").

`rewrite` turns follow-ups ("and for interns?") into standalone questions using the
conversation so far; the checkpointer keeps that conversation per session.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from time import monotonic

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime

from .config import settings
from .store import SessionStore

REWRITE_PROMPT = (
    "Rewrite the user's last message as a standalone question, using the conversation "
    "for any missing context. Reply with the question only."
)

ROUTE_PROMPT = """The user has uploaded these documents and is chatting with an assistant about them: {files}
Reply with one word:
- 'summary' if they want an overview of a whole document or of the person, company or topic it is about.
- 'docs' if they ask for specific facts, numbers, names, dates or rules, even if those might not be in the documents.
- 'chat' only for greetings, thanks, or questions about the assistant itself.
A message addressed to someone the documents are about ("tell me about yourself, Smit" with Smit's CV
uploaded) is about the documents, not the assistant. If unsure, reply 'docs'."""

ANSWER_PROMPT = """You answer questions using only facts stated in the document excerpts below.
- Never add facts from your own knowledge, even when you know the answer. If an excerpt only
  mentions or asks about a topic (a list of interview questions, say) without explaining it,
  say the documents don't explain it.
- Cite the excerpts you used with their numbers in square brackets, like [1] or [2][3].
- If the excerpts don't contain the answer, say so plainly instead of guessing.
- Keep answers short and clear; use bullet points for lists.

Excerpts:
{excerpts}"""

CHAT_PROMPT = (
    "You are DocChat, an assistant that answers questions about the user's uploaded "
    "documents. Be brief and friendly. You can't see what the documents say here, so never "
    "describe, summarise or guess their contents; suggest asking a question about them instead. "
    "If no documents are uploaded yet, invite the user to upload a PDF, TXT or Markdown file."
)

NOT_FOUND = (
    "I couldn't find that in your documents. Try rephrasing the question, "
    "or upload a document that covers it."
)

HISTORY_TURNS = 6

# Some models cite with full-width brackets (【1】); normalise them to [1].
CITATION = re.compile(r"[\[【](\d+)[\]】]")


@dataclass
class Context:
    session_id: str
    store: SessionStore


class ChatState(MessagesState):
    question: str
    route: str
    sources: list[dict]


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    # Extra retries ride out the free tier's tokens-per-minute limit (the SDK honours retry-after).
    return ChatGroq(model=settings.groq_model, temperature=0, max_retries=5)


def rewrite(state: ChatState) -> dict:
    latest = state["messages"][-1].content
    history = state["messages"][:-1][-HISTORY_TURNS:]
    if not history:
        return {"question": latest}
    reply = get_llm().invoke([SystemMessage(REWRITE_PROMPT), *history, HumanMessage(latest)])
    return {"question": reply.content.strip() or latest}


def route(state: ChatState, runtime: Runtime[Context]) -> dict:
    files = runtime.context.store.files(runtime.context.session_id)
    if not files:
        return {"route": "chat"}
    prompt = ROUTE_PROMPT.format(files=", ".join(files))
    reply = get_llm().invoke([SystemMessage(prompt), HumanMessage(state["question"])]).content.strip().lower()
    return {"route": next((r for r in ("summary", "chat") if reply.startswith(r)), "docs")}


def retrieve(state: ChatState, runtime: Runtime[Context]) -> dict:
    store, session_id = runtime.context.store, runtime.context.session_id
    if state["route"] == "summary":
        # "Summarise this" resembles no chunk in particular, so similarity search would miss.
        hits = [(doc, None) for doc in store.overview(session_id, settings.summary_chunks)]
    else:
        hits = [(doc, round(score, 3)) for doc, score in store.search(session_id, state["question"], settings.top_k)]
        # MIN_SCORE decides whether the documents cover the question at all. Weaker neighbours
        # of a good match still go to the model: in a CV, "Where did Smit study?" matches the
        # header with the name best, while the line naming the college scores well below it.
        if not hits or hits[0][1] < settings.min_score:
            hits = []
    sources = [
        {
            "n": number,
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "text": doc.page_content,
            "score": score,
        }
        for number, (doc, score) in enumerate(hits, start=1)
    ]
    return {"sources": sources}


def answer(state: ChatState) -> dict:
    excerpts = "\n\n".join(
        f"[{s['n']}] ({s['source']}, page {s['page']})\n{s['text']}" for s in state["sources"]
    )
    history = state["messages"][:-1][-HISTORY_TURNS:]
    reply = get_llm().invoke(
        [
            SystemMessage(ANSWER_PROMPT.format(excerpts=excerpts)),
            *history,
            HumanMessage(state["question"]),
        ]
    )
    text = CITATION.sub(r"[\1]", reply.content.strip())
    cited = {int(n) for n in CITATION.findall(text)}
    # Show only the excerpts the answer cites; an answer citing none is a "not in the excerpts" reply.
    sources = [s for s in state["sources"] if s["n"] in cited]
    return {"messages": [AIMessage(text)], "sources": sources}


def not_found(state: ChatState) -> dict:
    return {"messages": [AIMessage(NOT_FOUND)]}


def chat(state: ChatState, runtime: Runtime[Context]) -> dict:
    # Without this the model can claim nothing is uploaded when a question was routed here.
    files = runtime.context.store.files(runtime.context.session_id)
    uploaded = f"Uploaded documents: {', '.join(files)}." if files else "No documents are uploaded yet."
    history = state["messages"][-HISTORY_TURNS:]
    reply = get_llm().invoke([SystemMessage(f"{CHAT_PROMPT}\n{uploaded}"), *history])
    return {"messages": [AIMessage(reply.content.strip())], "sources": []}


def build_graph(checkpointer=None):
    builder = StateGraph(ChatState, context_schema=Context)
    builder.add_node("rewrite", rewrite)
    builder.add_node("route", route)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_node("not_found", not_found)
    builder.add_node("chat", chat)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "route")
    builder.add_conditional_edges(
        "route", lambda s: s["route"], {"docs": "retrieve", "summary": "retrieve", "chat": "chat"}
    )
    builder.add_conditional_edges(
        "retrieve", lambda s: "answer" if s["sources"] else "not_found", ["answer", "not_found"]
    )
    for node in ("answer", "not_found", "chat"):
        builder.add_edge(node, END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


class DocChat:
    """Ties the graph to the per-session vector stores."""

    def __init__(
        self,
        store: SessionStore,
        ttl_seconds: float = settings.session_ttl_minutes * 60,
        max_sessions: int = settings.max_sessions,
    ) -> None:
        self.store = store
        self.checkpointer = InMemorySaver()
        self.graph = build_graph(self.checkpointer)
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._last_used: dict[str, float] = {}  # least recently used first
        self._lock = Lock()

    def touch(self, session_id: str) -> None:
        """Marks a session as active and forgets idle or excess ones, so memory stays bounded."""
        now = monotonic()
        with self._lock:
            self._last_used.pop(session_id, None)
            self._last_used[session_id] = now
            stale = []
            for other, last_used in self._last_used.items():
                too_many = len(self._last_used) - len(stale) > self.max_sessions
                if other == session_id or not (too_many or now - last_used > self.ttl_seconds):
                    break
                stale.append(other)
            for other in stale:
                del self._last_used[other]
                self._drop(other)

    def ask(self, session_id: str, message: str) -> dict:
        self.touch(session_id)
        result = self.graph.invoke(
            {"messages": [HumanMessage(message)], "sources": []},
            config={"configurable": {"thread_id": session_id}},
            context=Context(session_id=session_id, store=self.store),
        )
        return {
            "answer": result["messages"][-1].content,
            "sources": result.get("sources", []),
            "route": result.get("route"),
        }

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._last_used.pop(session_id, None)
            self._drop(session_id)

    def _drop(self, session_id: str) -> None:
        self.store.clear(session_id)
        self.checkpointer.delete_thread(session_id)
