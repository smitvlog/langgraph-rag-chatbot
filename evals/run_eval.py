"""Evaluates DocChat on four kinds of document with LangSmith.

Each case loads one document into a fresh session, asks one question and scores:
- behaviour: it answers when the document covers the question and refuses when it doesn't
  (including documents that only *mention* a topic, like a list of interview questions)
- facts: an answer contains the key facts from the document (cases that list some)

Run it with GROQ_API_KEY and LANGSMITH_API_KEY set (in .env):

    python -m evals.run_eval

Results appear in LangSmith under Datasets & Experiments → docchat-eval.
"""

import re
import time
import uuid
from functools import cache
from pathlib import Path

from langsmith import Client

from app.graph import DocChat
from app.ingest import load_file, split
from app.store import SessionStore

ROOT = Path(__file__).resolve().parent.parent
DATASET = "docchat-eval"
# Groq's free tier allows about 8,000 tokens a minute and one question uses 2,000-3,000.
DELAY_SECONDS = 15

HANDBOOK = "samples/northwind-employee-handbook.pdf"
CV = "samples/priya-nair-cv.txt"
QUESTIONS = "samples/ml-interview-questions.md"
README = "README.md"

# (document, question, expected behaviour, facts an answer must contain)
CASES = [
    (HANDBOOK, "How many days of annual leave do I get?", "answer", ["24"]),
    (HANDBOOK, "What happens in my first week?", "answer", ["laptop"]),
    (HANDBOOK, "What is the daily meal limit when travelling?", "answer", ["45"]),
    (HANDBOOK, "What is the CEO's salary?", "refuse", []),
    (CV, "Tell me about yourself, Priya", "answer", ["Django"]),
    (CV, "Where did Priya study?", "answer", ["Pune Institute of Technology"]),
    (CV, "Which companies has Priya worked at?", "answer", ["Cloudnest", "Brightlane"]),
    (CV, "What is Priya's salary expectation?", "refuse", []),
    (QUESTIONS, "What is this document about?", "answer", ["interview"]),
    (QUESTIONS, "Which questions are about LangGraph?", "answer", ["LangGraph"]),
    (QUESTIONS, "What is overfitting and how do you prevent it?", "refuse", []),
    (QUESTIONS, "What is the difference between a list and a tuple?", "refuse", []),
    (README, "How do I run it locally?", "answer", ["uvicorn"]),
    (README, "Which embedding model does it use?", "answer", ["MiniLM"]),
    (README, "How do I run the tests?", "answer", ["pytest"]),
    (README, "What is the capital of France?", "refuse", []),
]

# "I couldn't find that…", "The excerpts don't explain…", "…is not mentioned".
REFUSAL = re.compile(
    r"couldn.t find|could not find"
    r"|(do|does|did)(n.t| not) (contain|provide|explain|mention|include|say|cover|give)"
    r"|not (mentioned|covered|stated|provided|included|explained)",
    re.IGNORECASE,
)

docchat = DocChat(SessionStore())


@cache
def chunks(document: str):
    path = ROOT / document
    return split(load_file(path.name, path.read_bytes()))


def target(inputs: dict) -> dict:
    time.sleep(DELAY_SECONDS)
    session_id = uuid.uuid4().hex
    docchat.store.add(session_id, Path(inputs["document"]).name, chunks(inputs["document"]))
    try:
        result = docchat.ask(session_id, inputs["question"])
    finally:
        docchat.forget(session_id)
    return {"answer": result["answer"], "route": result["route"], "sources": len(result["sources"])}


def refused(outputs: dict) -> bool:
    # DocChat shows no sources when it declines; the wording check catches a refusal the
    # model phrased as an ordinary answer.
    return outputs["sources"] == 0 or bool(REFUSAL.search(outputs["answer"]))


def behaviour(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["expect"]
    got = "refuse" if refused(outputs) else "answer"
    return {"key": "behaviour", "score": got == expected, "comment": f"expected {expected}, got {got}"}


def facts(outputs: dict, reference_outputs: dict) -> dict:
    required = reference_outputs["must_contain"]
    if not required:
        return {"key": "facts", "score": None, "comment": "nothing to check"}
    missing = [fact for fact in required if fact.lower() not in outputs["answer"].lower()]
    return {"key": "facts", "score": not missing, "comment": f"missing: {missing}" if missing else "all present"}


def sync_dataset(client: Client) -> None:
    """Makes the LangSmith dataset match CASES, keeping its experiment history."""
    if not client.has_dataset(dataset_name=DATASET):
        client.create_dataset(DATASET, description="DocChat: answer or refuse, on four kinds of document")
    wanted = {(doc, question): (expect, must) for doc, question, expect, must in CASES}
    existing = {}
    for example in client.list_examples(dataset_name=DATASET):
        key = (example.inputs["document"], example.inputs["question"])
        if key in wanted and key not in existing:
            existing[key] = example
        else:
            client.delete_example(example.id)
    new = [
        {"inputs": {"document": doc, "question": q}, "outputs": {"expect": expect, "must_contain": must}}
        for (doc, q), (expect, must) in wanted.items()
        if (doc, q) not in existing
    ]
    if new:
        client.create_examples(dataset_name=DATASET, examples=new)


def main() -> None:
    client = Client()
    sync_dataset(client)
    results = client.evaluate(
        target,
        data=DATASET,
        evaluators=[behaviour, facts],
        experiment_prefix="docchat",
        max_concurrency=1,
    )
    passed = total = 0
    for row in results:
        scores = {r.key: r for r in row["evaluation_results"]["results"]}
        ok = scores["behaviour"].score and scores["facts"].score is not False
        passed += ok
        total += 1
        print(f"{'PASS' if ok else 'FAIL'}  {row['example'].inputs['question']}  ({scores['behaviour'].comment}; {scores['facts'].comment})")
    print(f"\n{passed}/{total} cases passed. Experiment: {results.experiment_name}")


if __name__ == "__main__":
    main()
