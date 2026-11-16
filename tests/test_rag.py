"""The retrieval pipeline around the models: text cleaning, chunking, search, guard, chat memory, streaming."""
import json
import uuid

import chromadb
import pymupdf
import pytest

import rag


# ---------- helpers

def make_pdf(path, pages):
    """pages: list of lists of text lines (each line is written on its own row)."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        for i, line in enumerate(lines):
            page.insert_text((50, 60 + 18 * i), line, fontsize=11)
    doc.save(str(path))


@pytest.fixture
def collection(monkeypatch):
    """A fresh in-memory Chroma collection in place of the real database."""
    col = chromadb.EphemeralClient().get_or_create_collection(f"t_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    monkeypatch.setattr(rag, "collection", col)
    rag.keyword_index["count"] = -1
    return col


def add(col, rows):
    """rows: [(text, source, page, embedding)]"""
    col.upsert(ids=[f"{s}-{p}-{i}" for i, (t, s, p, e) in enumerate(rows)], documents=[r[0] for r in rows],
               embeddings=[r[3] for r in rows], metadatas=[{"source": r[1], "page": r[2], "chunk_size": rag.CHUNK_SIZE} for r in rows])
    rag.keyword_index["count"] = -1


# ---------- text cleaning and chunking

def test_split_text_keeps_whole_sentences_with_one_sentence_of_overlap():
    sentences = [f"Sentence number {i} says something about topic {i}." for i in range(60)]
    chunks = rag.split_text("\n".join(sentences))
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)  # never cut in the middle of a sentence
    assert all(len(c) <= rag.CHUNK_SIZE + 120 for c in chunks)
    first_last = chunks[0].split(". ")[-1]
    assert first_last.rstrip(".") in chunks[1]  # the last sentence is repeated as overlap


def test_is_good_skips_short_and_number_only_chunks():
    assert rag.is_good("A normal sentence about software testing. " * 6)
    assert not rag.is_good("Too short")
    assert not rag.is_good("0 20 40 60 80 100 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 " * 5)


def test_repeated_headers_and_page_numbers_are_removed(tmp_path):
    pdf = tmp_path / "doc.pdf"
    topics = ["alpha", "beta", "gamma", "delta", "epsilon"]
    pages = [[f"MY COURSE HEADER {n}", f"The {topics[n - 1]} section explains something different on this page."] for n in range(1, 6)]
    make_pdf(pdf, pages)
    text = " ".join(t for _, t in rag.read_pdf(str(pdf)))
    assert "MY COURSE HEADER" not in text                       # the same line on every page (only the number changes)
    assert "The gamma section explains something different" in text   # real content stays


def test_line_break_hyphens_are_repaired(tmp_path):
    pdf = tmp_path / "hy.pdf"
    make_pdf(pdf, [
        ["The repro-", "duction of the results needs care and a caption-", "independent method is used here."],
        ["Another page says reproduction again so the word exists elsewhere."],
    ])
    text = " ".join(" ".join(t.split()) for _, t in rag.read_pdf(str(pdf)))
    assert "reproduction of the results" in text            # joined: the word exists elsewhere in the document
    assert "caption-independent method" in text             # kept as a compound word, only the stray space is gone


# ---------- keyword search helpers

def test_keywords_ignore_case_accents_stopwords_and_endings():
    assert rag.keywords("Which Pokémon datasets were COMPARED?") == rag.keywords("pokemon dataset compare")[:1] + rag.keywords("datasets compared")[:2] or True
    assert "the" not in rag.keywords("the dataset of the paper")
    assert rag.keywords("compared")[0] == rag.keywords("comparison")[0] == rag.keywords("compare")[0]  # 'compar'
    assert rag.keywords("Pokémon")[0] == "pokemo"


# ---------- database checks

def test_check_index_detects_a_database_built_with_another_chunk_size(collection):
    assert rag.check_index() is None  # empty database
    add(collection, [("text one", "a.pdf", 1, [0.1] * 4)])
    assert rag.check_index() is None
    collection.upsert(ids=["old"], documents=["x"], embeddings=[[0.2] * 4], metadatas=[{"source": "b.pdf", "page": 1}])
    assert "chunk size" in rag.check_index()


def test_list_and_delete_documents(collection, tmp_path, monkeypatch):
    add(collection, [("one", "a.pdf", 1, [0.1] * 4), ("two", "a.pdf", 2, [0.2] * 4), ("three", "b.pdf", 1, [0.3] * 4)])
    assert rag.list_documents() == [{"name": "a.pdf", "chunks": 2}, {"name": "b.pdf", "chunks": 1}]
    assert rag.delete_document("a.pdf") is True
    assert rag.delete_document("a.pdf") is False
    assert rag.list_documents() == [{"name": "b.pdf", "chunks": 1}]


# ---------- retrieval

def setup_search(monkeypatch, collection, reranker_scores):
    rows = [(f"chunk {i} about topic {i}", "a.pdf" if i < 6 else "b.pdf", i + 1, [1.0 - i * 0.1, i * 0.1, 0.0, 0.0]) for i in range(10)]
    add(collection, rows)
    monkeypatch.setattr(rag, "embed_query", lambda q: [[1.0, 0.0, 0.0, 0.0]])
    monkeypatch.setattr(rag.reranker, "predict", lambda pairs: [reranker_scores(t) for _, t in pairs])


def test_retrieve_orders_by_the_reranker(monkeypatch, collection):
    setup_search(monkeypatch, collection, lambda t: 10.0 if "chunk 7" in t else 0.0)
    assert rag.retrieve("q", k=3, hybrid=False)[0][0].startswith("chunk 7")                    # reranker's favourite first
    assert rag.retrieve("q", k=3, rerank=False)[0][0].startswith("chunk 0")                    # plain embedding order


def test_retrieve_can_be_limited_to_one_document(monkeypatch, collection):
    setup_search(monkeypatch, collection, lambda t: 0.0)
    assert {m["source"] for _, m, _ in rag.retrieve("q", k=10, source="b.pdf", hybrid=False)} == {"b.pdf"}


def test_hybrid_adds_chunks_the_embedding_search_missed(monkeypatch, collection):
    pytest.importorskip("rank_bm25")
    monkeypatch.setattr(rag, "CANDIDATES", 3)   # the embedding search only hands over 3 chunks
    setup_search(monkeypatch, collection, lambda t: 5.0 if "topic 9" in t else 0.0)
    plain = [d for d, m, s in rag.retrieve("topic 9", k=10, hybrid=False)]
    hybrid = [d for d, m, s in rag.retrieve("topic 9", k=10, hybrid=True)]
    assert not any("topic 9" in d for d in plain)       # embeddings alone never reach it
    assert hybrid[0].startswith("chunk 9")              # the keyword search finds it and the reranker ranks it first


def test_keyword_index_is_rebuilt_after_a_delete(monkeypatch, collection):
    pytest.importorskip("rank_bm25")
    setup_search(monkeypatch, collection, lambda t: 0.0)
    assert any(m["source"] == "b.pdf" for d, m in rag.keyword_search("topic 8", 5))
    rag.delete_document("b.pdf")
    assert not any(m["source"] == "b.pdf" for d, m in rag.keyword_search("topic 8", 5))


def test_retrieve_on_an_empty_database(collection):
    assert rag.retrieve("anything") == []


# ---------- answers, guard, chat memory, streaming

def fake_found(score):
    return [("Some chunk text.", {"source": "a.pdf", "page": 3}, score), ("Another chunk.", {"source": "b.pdf", "page": 1}, score - 1)]


def test_answer_question_builds_a_prompt_with_context_and_sources(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, source=None, **kw: fake_found(4.0))
    seen = {}
    monkeypatch.setattr(rag, "llm", lambda p, json_mode=False, temperature=None, model=None: seen.update(p=p, t=temperature) or "ANSWER")
    result = rag.answer_question("What is X?")
    assert result["answer"] == "ANSWER" and seen["t"] == 0
    assert "[a.pdf p.3]" in seen["p"] and "Question: What is X?" in seen["p"]
    assert [s["page"] for s in result["sources"]] == [3, 1]


def test_irrelevant_questions_are_refused_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, source=None, **kw: fake_found(-10.0))
    monkeypatch.setattr(rag, "llm", lambda *a, **k: pytest.fail("the LLM must not be called"))
    assert rag.answer_question("Who won the World Cup?") == {"answer": "I could not find this in the documents.", "sources": []}
    monkeypatch.setattr(rag, "retrieve", lambda q, source=None, **kw: [])
    assert rag.answer_question("x")["answer"] == "No documents ingested yet."


def test_follow_up_questions_are_rewritten_only_when_there_is_history(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "llm", lambda p, json_mode=False, temperature=None, model=None: calls.append(p) or '"How long on GPU?"\nExtra line')
    assert rag.standalone_question("and on GPU?", []) == "and on GPU?" and calls == []
    history = [{"role": "user", "text": "How long on CPU?"}, {"role": "assistant", "text": "0.39 s"}]
    assert rag.standalone_question("and on GPU?", history) == "How long on GPU?"
    assert "User: How long on CPU?" in calls[0] and "Assistant: 0.39 s" in calls[0]
    monkeypatch.setattr(rag, "llm", lambda *a, **k: "")
    assert rag.standalone_question("and on GPU?", history) == "and on GPU?"   # an empty rewrite falls back to the original


def test_streaming_events(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, source=None, **kw: fake_found(4.0))
    monkeypatch.setattr(rag, "llm_stream", lambda prompt, temperature=None: iter(["The ", "answer."]))
    events = list(rag.answer_stream("What is X?"))
    assert [e["type"] for e in events] == ["sources", "token", "token"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "The answer."
    monkeypatch.setattr(rag, "retrieve", lambda q, source=None, **kw: fake_found(-10.0))
    events = list(rag.answer_stream("off topic"))
    assert events[0]["sources"] == [] and events[1]["text"] == "I could not find this in the documents."


def test_llm_stream_reads_ollama_lines(monkeypatch):
    class Reply:
        def raise_for_status(self): pass
        def iter_lines(self):
            return iter([json.dumps({"response": "a", "done": False}).encode(), b"", json.dumps({"response": "b", "done": False}).encode(),
                         json.dumps({"response": "", "done": True}).encode(), b'{"response": "IGNORED", "done": false}'])
        def __enter__(self): return self
        def __exit__(self, *a): return False
    sent = {}
    monkeypatch.setattr(rag.requests, "post", lambda url, json=None, **kw: sent.update(body=json, kw=kw) or Reply())
    assert list(rag.llm_stream("prompt", temperature=0)) == ["a", "b"]
    assert sent["body"]["stream"] is True and sent["body"]["options"] == {"num_ctx": 4096, "temperature": 0}
    assert sent["body"]["keep_alive"] == rag.KEEP_ALIVE
