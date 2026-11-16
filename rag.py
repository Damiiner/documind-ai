
import json
import os
import re
import sys
import unicodedata

import chromadb
import requests
import pymupdf
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # was all-MiniLM-L6-v2 (baseline hit@3 = 58%)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge: queries only, not documents
LLM_MODEL = "llama3.2:3b"
KEEP_ALIVE = "30m"  # keep the model in memory between questions (Ollama unloads it after 5 minutes by default)
OLLAMA_URL = "http://localhost:11434/api/generate"
CHUNK_SIZE = 500      # characters, chunks end on sentence boundaries (answer eval: 500 / 6 gave 25 good answers of 27, 900 + hybrid gave 22)
TOP_K = 6             # chunks sent to the LLM (a 3B model answers better from a short context)
CANDIDATES = 30       # chunks the embedding search hands to the reranker
KEYWORD_CANDIDATES = 20  # chunks the keyword search (BM25) adds to that pool
HYBRID = False        # True adds keyword candidates (needs rank-bm25). It raised retrieval at chunk size 900 but did not improve the answers.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUIZ_MODEL = "llama3.2:3b"  # quiz writing is harder than answering: try "qwen2.5:7b" if your RAM allows
MIN_SCORE = -8  # (eval: in-scope >= -6.33, out-of-scope <= -10.16) if even the best chunk scores below this, the question is not about the documents

embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def remove_headers(raw):
    """Drop short lines that repeat on at least half of the pages (running headers, footers, page numbers).
    They add the same words to every chunk and make unrelated chunks look similar."""
    def key(line):
        return re.sub(r"\d+", "#", " ".join(line.lower().split()))

    seen = {}
    for _, text in raw:
        for k in {key(l) for l in text.split("\n") if 0 < len(l.strip()) <= 80}:
            seen[k] = seen.get(k, 0) + 1
    common = {k for k, count in seen.items() if count >= max(3, len(raw) / 2)}
    return [(p, "\n".join(l for l in text.split("\n") if key(l) not in common)) for p, text in raw]


def read_pdf(path):
    """Return a list of (page_number, text). PyMuPDF keeps the spaces between words.
    Repeated headers/footers are removed. Line-break hyphens ('repro-' + 'duction') are repaired:
    the word is joined if it exists elsewhere in the document, otherwise the hyphen is kept."""
    with pymupdf.open(path) as doc:  # closed right away, so the file can be deleted later (Windows locks open files)
        raw = remove_headers([(i + 1, page.get_text()) for i, page in enumerate(doc)])
    vocab = set(re.findall(r"[a-z]+", " ".join(t for _, t in raw).lower()))

    def join(m):
        word = m.group(1) + m.group(2)
        return word if word.lower() in vocab else m.group(1) + "-" + m.group(2)

    return [(p, re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", join, t)) for p, t in raw if t.strip()]


def split_text(text):
    """Group whole sentences into chunks; repeat the last sentence as overlap."""
    text = " ".join(text.split())  # remove newlines and extra spaces
    sentences = re.split(r"(?<=[.!?]) ", text)
    chunks, current, last = [], "", ""
    for s in sentences:
        if len(current) + len(s) > CHUNK_SIZE and current:
            chunks.append(current)
            current = last + " " + s
        else:
            current = (current + " " + s).strip()
        last = s
    if current:
        chunks.append(current)
    return chunks


def is_good(chunk):
    """Skip tiny chunks and junk (figure labels, axis numbers, tables)."""
    letters = sum(c.isalpha() or c == " " for c in chunk)
    return len(chunk) >= 150 and letters / len(chunk) > 0.7


def ingest(path):
    filename = os.path.basename(path)
    collection.delete(where={"source": filename})  # re-upload replaces the old version
    keyword_index["count"] = -1  # the keyword index must be rebuilt
    ids, texts, metas = [], [], []
    for page, text in read_pdf(path):
        for j, chunk in enumerate(split_text(text)):
            if not is_good(chunk):
                continue
            ids.append(f"{filename}-p{page}-{j}")
            texts.append(chunk)
            metas.append({"source": filename, "page": page, "chunk_size": CHUNK_SIZE})
    if not texts:
        print("No text found (scanned PDF? OCR needed).")
        return 0
    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    print(f"Ingested {len(texts)} chunks from {filename}")
    return len(texts)


def check_index():
    """Return a warning text if the stored chunks were built with another chunk size than CHUNK_SIZE, else None."""
    metas = collection.get(include=["metadatas"])["metadatas"]
    if not metas:
        return None
    sizes = {m.get("chunk_size") for m in metas}
    if sizes != {CHUNK_SIZE}:
        found = ", ".join("unknown" if s is None else str(s) for s in sorted(sizes, key=str))
        return (f"the database holds chunks built with chunk size {found}, but rag.py says {CHUNK_SIZE}. "
                "Delete the db folder and ingest all documents again.")
    return None


def list_documents():
    """Return [{"name": file name, "chunks": number of chunks}]."""
    counts = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]


def delete_document(name):
    """Remove a document's chunks and its uploaded file. Returns False if it does not exist."""
    if not collection.get(where={"source": name}, include=["metadatas"])["ids"]:
        return False
    collection.delete(where={"source": name})
    keyword_index["count"] = -1  # the keyword index must be rebuilt
    path = os.path.join("uploads", name)
    if os.path.exists(path):
        os.remove(path)
    return True


def embed_query(question):
    return embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()


STOPWORDS = set("the a an of to in is are and or what which how does do for on with by that this it as at be from was were can".split())
keyword_index = {"count": -1, "model": None, "docs": [], "metas": []}


def keywords(text):
    """Lowercase words without accents or stopwords, cut to 6 letters so 'compare/compared/comparison' match."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return [w[:6] for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS]


def build_keyword_index():
    """BM25 keyword index over all chunks, rebuilt only when documents were added or removed."""
    from rank_bm25 import BM25Okapi  # imported here so the app also runs without it when HYBRID is False

    count = collection.count()
    if keyword_index["count"] != count:
        data = collection.get(include=["documents", "metadatas"])
        docs, metas = data["documents"], data["metadatas"]
        keyword_index.update(count=count, docs=docs, metas=metas,
                             model=BM25Okapi([keywords(d) for d in docs]) if docs else None)
    return keyword_index


def keyword_search(question, n, source=None):
    """The n best chunks by shared words with the question: [(text, metadata)]. Finds exact terms embeddings can miss."""
    idx = build_keyword_index()
    if idx["model"] is None:
        return []
    scores = idx["model"].get_scores(keywords(question))
    found = []
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if scores[i] <= 0 or len(found) == n:
            break
        if source is None or idx["metas"][i]["source"] == source:
            found.append((idx["docs"][i], idx["metas"][i]))
    return found


def retrieve(question, k=TOP_K, rerank=True, source=None, hybrid=None):
    """Find candidate chunks, then re-sort them with the reranker (best chunk first).
    rerank=False: embedding search only.
    hybrid=True (default: the HYBRID setting): the embedding candidates are joined by keyword-search candidates, so a chunk that
    contains the exact words of the question (a section title, a name) is not lost.
    The score returned with each chunk is the reranker score (or minus the distance without the reranker)."""
    if hybrid is None:
        hybrid = HYBRID
    where = {"source": source} if source else None
    res = collection.query(query_embeddings=embed_query(question), n_results=CANDIDATES, where=where)
    docs, metas = res["documents"][0], res["metadatas"][0]
    if not docs:
        return []
    if not rerank:
        return [(d, m, -dist) for d, m, dist in zip(docs, metas, res["distances"][0])][:k]

    pool = dict(zip(docs, metas))  # text -> metadata (also removes duplicates)
    if hybrid:
        for d, m in keyword_search(question, KEYWORD_CANDIDATES, source):
            pool.setdefault(d, m)
    texts = list(pool)
    scores = reranker.predict([(question, t) for t in texts])
    best_first = sorted(range(len(texts)), key=lambda i: -scores[i])
    return [(texts[i], pool[texts[i]], float(scores[i])) for i in best_first[:k]]


def llm(prompt, json_mode=False, temperature=None, model=None):
    """Send a prompt to Ollama and return the text of the reply."""
    body = {"model": model or LLM_MODEL, "prompt": prompt, "stream": False, "keep_alive": KEEP_ALIVE}
    if json_mode:
        body["format"] = "json"  # forces valid JSON output
    body["options"] = {"num_ctx": 4096}  # room for 6 chunks + question (Ollama's default window can be too small)
    if temperature is not None:
        body["options"]["temperature"] = temperature
    r = requests.post(OLLAMA_URL, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


def llm_stream(prompt, temperature=None):
    """Like llm(), but yields the reply piece by piece while Ollama is still writing it."""
    body = {"model": LLM_MODEL, "prompt": prompt, "stream": True, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": 4096}}
    if temperature is not None:
        body["options"]["temperature"] = temperature
    with requests.post(OLLAMA_URL, json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break


REWRITE_PROMPT = (
    "Rewrite the last question so that it can be understood without the conversation. "
    "Replace pronouns and vague references (it, that, they, this method) with what they refer to. "
    "If the question is already clear on its own, repeat it unchanged. Reply with the question only.\n\n"
    "Conversation:\n{history}\n\nLast question: {question}\n\nStandalone question:"
)


def standalone_question(question, history):
    """Chat memory: turn a follow-up such as 'and for SD3.5?' into a question that makes sense alone.
    history is the earlier turns [{"role": "user" or "assistant", "text": ...}]. With no history nothing changes."""
    if not history:
        return question
    turns = history[-4:]
    text = "\n".join(f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['text'][:500]}" for t in turns)
    lines = llm(REWRITE_PROMPT.format(history=text, question=question), temperature=0).strip().splitlines()
    rewritten = lines[0].strip().strip('"') if lines else ""
    return rewritten if 0 < len(rewritten) <= 300 else question  # keep the original if the rewrite looks wrong


def get_chunks(source):
    """All chunks of one document as [(text, page)] (used for quiz generation)."""
    data = collection.get(where={"source": source}, include=["documents", "metadatas"])
    return [(d, m["page"]) for d, m in zip(data["documents"], data["metadatas"])]


def prepare(question, source=None):
    """Retrieve the chunks and build the prompt. Returns {"refusal": text} or {"prompt": text, "sources": [...]}."""
    found = retrieve(question, source=source)
    if not found:
        return {"refusal": "No documents ingested yet."}
    if max(score for _, _, score in found) < MIN_SCORE:  # nothing relevant: refuse without calling the LLM (no hallucination)
        return {"refusal": "I could not find this in the documents."}

    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}" for d, m, sc in found
    )
    prompt = (
        "You are a study assistant. Answer the question using the context below. "
        "The context may contain only part of the answer, so use whatever is relevant. "
        "Only if nothing in the context is relevant, say \"I could not find this in the documents.\" "
        "Answer in 2-4 sentences.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    sources = [
        {"source": m["source"], "page": m["page"], "text": d[:300], "score": round(sc, 2)}
        for d, m, sc in found
    ]
    return {"prompt": prompt, "sources": sources}


def answer_question(question, source=None):
    """Return {"answer": str, "sources": [{"source", "page", "text", "score"}]} (used by the CLI and the evals)."""
    p = prepare(question, source)
    if "refusal" in p:
        return {"answer": p["refusal"], "sources": []}
    answer = llm(p["prompt"], temperature=0)  # same question, same answer (the default temperature made runs differ)
    return {"answer": answer, "sources": p["sources"]}


def answer_stream(question, source=None):
    """Like answer_question(), but yields events: first {"type": "sources"}, then many {"type": "token"}."""
    p = prepare(question, source)
    yield {"type": "sources", "sources": p.get("sources", [])}
    if "refusal" in p:
        yield {"type": "token", "text": p["refusal"]}
        return
    for piece in llm_stream(p["prompt"], temperature=0):
        yield {"type": "token", "text": piece}


def ask(question):
    result = answer_question(question)
    print("\nRetrieved chunks:")
    for src in result["sources"]:
        print(f"- p.{src['page']}  score={src['score']}  {src['text'][:100]!r}")
    print("\n" + result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- {src['source']}, page {src['page']}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("ingest", "ask"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "ingest":
        ingest(sys.argv[2])
    else:
        ask(" ".join(sys.argv[2:]))
