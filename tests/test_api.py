"""The HTTP API (FastAPI): uploads and their limits, documents, questions, streaming, quizzes."""
import json
import os

import pymupdf
import pytest
from fastapi.testclient import TestClient

import main
import rag

client = TestClient(main.app)


def pdf_bytes(pages=1):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), "hello world " * 20)
    return doc.tobytes()


def upload(name, data, content_type="application/pdf"):
    return client.post("/documents", files={"file": (name, data, content_type)})


# ---------- uploads

def test_a_valid_pdf_is_ingested(monkeypatch):
    monkeypatch.setattr(rag, "ingest", lambda path: 7)
    r = upload("ok.pdf", pdf_bytes(2))
    assert r.status_code == 200 and r.json() == {"filename": "ok.pdf", "pages": 2, "chunks": 7}


def test_only_real_pdfs_are_accepted(monkeypatch):
    monkeypatch.setattr(rag, "ingest", lambda path: 7)
    assert upload("notes.txt", b"hello", "text/plain").status_code == 400
    assert upload("fake.pdf", b"this is not a pdf").status_code == 400
    assert not os.path.exists(os.path.join("uploads", "fake.pdf"))  # rejected files are not kept


def test_size_and_page_limits(monkeypatch):
    monkeypatch.setattr(rag, "ingest", lambda path: 7)
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    assert upload("big.pdf", pdf_bytes()).status_code == 413
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 50)
    monkeypatch.setattr(main, "MAX_PAGES", 1)
    assert upload("long.pdf", pdf_bytes(3)).status_code == 413


def test_a_pdf_without_text_is_rejected(monkeypatch):
    monkeypatch.setattr(rag, "ingest", lambda path: 0)  # scanned PDF: nothing to read
    r = upload("scan.pdf", pdf_bytes())
    assert r.status_code == 422 and "OCR" in r.json()["detail"]
    assert not os.path.exists(os.path.join("uploads", "scan.pdf"))


def test_the_client_cannot_choose_where_the_file_is_saved(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag, "ingest", lambda path: seen.update(path=path) or 3)
    r = upload("../../evil.pdf", pdf_bytes())
    assert r.json()["filename"] == "evil.pdf" and os.path.dirname(seen["path"]) == "uploads"


# ---------- documents

def test_list_and_delete(monkeypatch):
    monkeypatch.setattr(rag, "list_documents", lambda: [{"name": "a.pdf", "chunks": 3}])
    assert client.get("/documents").json() == [{"name": "a.pdf", "chunks": 3}]
    monkeypatch.setattr(rag, "delete_document", lambda name: name == "a.pdf")
    assert client.delete("/documents/a.pdf").json() == {"deleted": "a.pdf"}
    assert client.delete("/documents/nope.pdf").status_code == 404


def test_the_react_dev_server_is_allowed_by_cors():
    r = client.options("/ask", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


# ---------- questions

@pytest.fixture
def fake_answers(monkeypatch):
    seen = {}
    monkeypatch.setattr(rag, "standalone_question", lambda q, history: seen.update(history=history) or (q if not history else "STANDALONE " + q))
    monkeypatch.setattr(rag, "answer_question", lambda q, source=None: {"answer": f"answer to {q}", "sources": []})

    def stream(q, source=None):
        yield {"type": "sources", "sources": [{"source": "a.pdf", "page": 1, "text": "t", "score": 1.0}]}
        for piece in ["Hello ", "world"]:
            yield {"type": "token", "text": piece}

    monkeypatch.setattr(rag, "answer_stream", stream)
    return seen


def test_ask_without_history_still_works(fake_answers):
    r = client.post("/ask", json={"question": "What is X?"}).json()
    assert r["answer"] == "answer to What is X?" and r["question_used"] == "What is X?" and fake_answers["history"] == []


def test_a_follow_up_is_rewritten_using_the_history(fake_answers):
    history = [{"role": "user", "text": "q1"}, {"role": "assistant", "text": "a1"}]
    r = client.post("/ask", json={"question": "and for SD3.5?", "history": history}).json()
    assert r["question_used"] == "STANDALONE and for SD3.5?"


def test_bad_history_is_rejected(fake_answers):
    turn = {"role": "user", "text": "x"}
    assert client.post("/ask", json={"question": "x", "history": [turn] * 11}).status_code == 422
    assert client.post("/ask", json={"question": "x", "history": [{"role": "user"}]}).status_code == 422


def test_the_answer_can_be_streamed(fake_answers):
    r = client.post("/ask_stream", json={"question": "and for SD3.5?", "history": [{"role": "user", "text": "q1"}]})
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.strip().split("\n")]
    assert [e["type"] for e in events] == ["question", "sources", "token", "token"]
    assert events[0]["text"] == "STANDALONE and for SD3.5?"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello world"


def test_an_error_while_streaming_becomes_an_error_event(monkeypatch, fake_answers):
    def broken(q, source=None):
        yield {"type": "sources", "sources": []}
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr(rag, "answer_stream", broken)
    events = [json.loads(line) for line in client.post("/ask_stream", json={"question": "x"}).text.strip().split("\n")]
    assert events[-1]["type"] == "error" and "Ollama is not running" in events[-1]["text"]


# ---------- quizzes

def test_quiz_endpoint(monkeypatch):
    monkeypatch.setattr(rag, "get_chunks", lambda source: [("text", 1)] if source == "a.pdf" else [])
    monkeypatch.setattr(main.quiz, "generate_quiz", lambda chunks, n, llm: [{"question": "Q?", "options": ["a", "b", "c", "d"], "answer": 0, "explanation": "e", "page": 1}])
    ok = client.post("/quiz", json={"source": "a.pdf", "n": 3})
    assert ok.status_code == 200 and ok.json()["questions"][0]["page"] == 1
    assert client.post("/quiz", json={"source": "nope.pdf"}).status_code == 404
    assert client.post("/quiz", json={"source": "a.pdf", "n": 0}).status_code == 422
    assert client.post("/quiz", json={"source": "a.pdf", "n": 11}).status_code == 422
    monkeypatch.setattr(main.quiz, "generate_quiz", lambda chunks, n, llm: [])
    assert client.post("/quiz", json={"source": "a.pdf"}).status_code == 502
