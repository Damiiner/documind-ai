# DocuMind AI

A study assistant for your own documents. Upload PDFs, ask questions and get answers with **page-level sources**, ask follow-up questions, or generate **multiple-choice quizzes** from a document. Everything runs locally: no API keys, no data leaves your computer.

> Add a short screen recording here (`docs/demo.gif`): upload a PDF, ask a question, ask a follow-up, generate a quiz.

## Features

- **Question answering with sources.** Every answer shows the chunks it was built from (file, page, text).
- **Refuses what the documents do not contain.** Off-topic questions get "I could not find this in the documents" instantly, without calling the LLM.
- **Chat memory.** Follow-ups such as "and on GPU?" are rewritten into standalone questions before searching.
- **Streaming answers.** The text appears while the model is still writing it.
- **One document or all.** Search everything or only the selected document.
- **Quiz generator.** Multiple-choice questions written from the document, each with a verbatim evidence sentence and its source page.
- **Safe uploads.** Size and page limits, real-PDF check, sanitised file names.

## How it works

```
PDF ──► PyMuPDF text ──► clean (headers/footers, line-break hyphens) ──► sentence-based chunks (500 chars)
                                                                              │
                                              bge-small-en-v1.5 embeddings ───┘──► ChromaDB (cosine)

question ──► (follow-up? rewrite with the last 4 messages)
        ──► embed ──► 30 nearest chunks ──► cross-encoder reranker ──► top 6
        ──► best score below threshold?  yes: refuse (no LLM call)
                                         no : prompt + 6 chunks ──► llama3.2:3b (Ollama, temperature 0) ──► streamed answer + sources
```

**Quiz generation:** one chunk → the LLM writes a question, the correct answer, three wrong answers and one evidence sentence (JSON) → Python rejects the question if it is vague ("the experiment", "Figure 3"), contains formulas, has circular or near-duplicate options, a unit mismatch, or if the evidence sentence is not **word for word** in the chunk → a second LLM call answers the question without seeing the key, and the question is dropped if the answers differ.

| Part | Choice |
|---|---|
| Text extraction | PyMuPDF (pypdf glued words together: 126 glued tokens vs 0 on the test paper) |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector store | ChromaDB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `llama3.2:3b` through Ollama (8 GB RAM, CPU only) |
| Backend | FastAPI (streaming as newline-delimited JSON) |
| Frontend | React + Vite |
| Optional | `rank-bm25` keyword search (`HYBRID = True` in `rag.py`) |

## Run it (Windows)

Requirements: Python 3.11 or newer (developed and tested on Windows with Python 3.14), Node.js, [Ollama](https://ollama.com).

```
ollama pull llama3.2:3b

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app
```

In a second terminal:

```
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The `frontend` folder is a Vite React project; the app lives in `src/App.jsx` and `src/main.jsx`.

Command line, without the web app:

```
python rag.py ingest paper.pdf
python rag.py ask "What is the main contribution?"
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /documents` | Upload a PDF (max 50 MB, 300 pages) |
| `GET /documents` | List documents with chunk counts |
| `DELETE /documents/{name}` | Remove a document |
| `POST /ask` | `{question, source?, history?}` → `{answer, sources, question_used}` |
| `POST /ask_stream` | Same, streamed: events `question`, `sources`, `token`, `error` |
| `POST /quiz` | `{source, n}` → questions with options, answer index, evidence, page |

Interactive docs: <http://127.0.0.1:8000/docs>.

## Tests and evaluation

```
pip install -r requirements-dev.txt
pytest                      # 41 tests, offline, about 2 seconds (models are replaced by stand-ins)

python eval.py              # retrieval benchmark, no LLM, about 1 minute
python eval_answers.py      # end-to-end answer benchmark, about 8-12 minutes
python diagnose.py          # what is really stored in the database
```

The evaluation scripts refuse to run if the database was built with a different chunk size than `rag.py` says.

### Benchmark

Three documents with different layouts: a 42-page math-heavy research paper (arXiv:2610.31659), a 9-slide lecture deck and a 3-page textbook chapter. 31 retrieval questions and 27 answer questions with known answers. Written and rated by one person, so treat it as a development benchmark, not a general result.

**Retrieval** (a hit means a retrieved chunk *contains the evidence text*; first / top 3 / top 6):

| Documents | Embeddings only | + reranker (final) |
|---|---|---|
| All 31 questions | 58 / 68 / 81 % | **68 / 81 / 87 %** |
| Research paper (16) | 25 / 38 / 62 % | 44 / 62 / 75 % |
| Slides (7) | 100 / 100 / 100 % | 100 / 100 / 100 % |
| Chapter (8) | 88 / 100 / 100 % | 88 / 100 / 100 % |

**Answers** (final configuration, all 27 questions, read by hand):

| | |
|---|---|
| Good answers | **23 of 27** |
| Partially correct | 2 |
| Wrong or incomplete | 2 |
| Numbers in answers that are not in the retrieved text | 0 of 27 |
| Off-topic questions refused | 3 of 3 (every run) |
| Time per answer (8 GB laptop, CPU only) | about 17 s including the wait before the first word |

The same configuration scored 25 good answers in an earlier run and a variant scored 22, so differences of 1 to 3 answers are within the run-to-run noise of this benchmark.

**Quiz quality** (10 questions per version, rated by hand): the first version produced about 2 usable questions out of 6; after the evidence and vagueness filters, 3 good / 2 weak / 4 flawed out of 9; after the formula, duplicate, circular and unit filters, **6 good / 4 weak / 0 flawed out of 10**.

## What I learned

1. **Page-level retrieval metrics hid real failures.** My first benchmark counted a hit when any chunk from the right page came back. The page holding the answer had several chunks, so the metric looked fine while the LLM never saw the sentence it needed. Switching to "the chunk contains the evidence text" showed the true numbers (the paper's top-6 hit rate dropped from 94 % to 75 %).
2. **Check that the thing you measure is the thing you think you measure.** For several experiments the database still held 500-character chunks while the code said 900, so I compared the wrong settings. The chunk size is now stored with every chunk and the evaluation scripts stop on a mismatch.
3. **Better retrieval did not mean better answers.** Chunks of 900 characters plus keyword search found the evidence more often, but a 3B model answered worse from the longer context (22 vs 25 good answers). I chose the configuration by answer quality, not by retrieval score.
4. **Calibrate refusals with data.** A relevance threshold of -5 looked sensible but would have refused a valid question (its best score was -6.2). The evaluation puts the lowest in-scope score (-6.3) and the highest off-topic score (-10.2) side by side, and the threshold sits between them at -8.
5. **For small models, rules beat prompting.** Telling the quiz model to write clear questions did little. Deterministic filters (verbatim evidence, vagueness, duplicates) removed the bad questions.
6. **Automatic checks are wrong in both directions.** Keyword checks missed correct answers ("accuracy" vs "accurate") and passed wrong ones. Every run logs all answers to a file so that a human reads them.
7. **Remove repeated page headers.** "INTRODUCTION TO CYBERSECURITY Slide 6" at the top of every slide made unrelated chunks look alike. Dropping lines that repeat on at least half of the pages fixed slide retrieval from 86 % to 100 %.

Things I tried that did not help, and are not in the final version: smaller chunks (500) with 8 chunks to the LLM, reciprocal rank fusion of the embedding and reranker orders (the reranker alone was better), and hybrid keyword search (it improved retrieval at 900 characters but not the answers; it stays available as `HYBRID = True`).

## Known limitations

- **Some questions fail.** The question "which datasets are used" is not answered because the chunk with the dataset names is not retrieved at 500 characters. "What do the authors say about using only one canary" gets a partly invented rationale. The login-steps question sometimes returns only the first step.
- **A 3B model misreads tables** (it said the paper gives no GPU timing; Table 3 does) and sometimes adds details that are not in the text.
- **Typeset equations** are extracted as symbol soup and are hard to retrieve.
- **No OCR:** scanned PDFs are rejected. Only PDFs, only English.
- **Small benchmark:** 3 documents, one rater, results vary by about 2 answers between runs.
- **Local, single user:** no authentication; not meant to be exposed to the internet.

## Possible next steps

OCR for scanned PDFs, `.docx` and `.txt` support, numbered citations inside the answer text, a larger model on a machine with more memory, a bigger benchmark with several raters, French support (multilingual embeddings).

## Project layout

```
rag.py             ingestion, retrieval, prompts, streaming, command line
quiz.py            quiz generation and its quality filters
main.py            FastAPI app
eval.py            retrieval benchmark
eval_answers.py    end-to-end answer benchmark
diagnose.py        database inspection
tests/             41 offline tests
frontend/          React + Vite app
```
