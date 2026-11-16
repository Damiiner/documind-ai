# DocuMind AI

A local study assistant for your own PDFs. Upload a document, ask questions and get answers with page references, or generate multiple-choice quizzes from it. It runs on your machine with Ollama, so no API key is needed and no document leaves the computer.

<!-- demo: docs/demo.gif -->

## Features

- Answers with sources: each answer lists the chunks it came from (file, page, text).
- Refuses questions the documents cannot answer, without calling the model.
- Follow-up questions ("and on GPU?") are rewritten into standalone questions using the previous messages.
- Streamed answers, and search across all documents or a single one.
- Quiz generator: each question comes with an evidence sentence copied from the document and its page.

## How it works

1. **Ingestion.** PyMuPDF extracts the text. Repeated headers and footers are removed, line-break hyphens are repaired, and the text is split into sentence-aligned chunks of about 500 characters. Chunks are embedded with `bge-small-en-v1.5` and stored in ChromaDB.
2. **Retrieval.** The question is embedded, the 30 nearest chunks are re-ranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and the top 6 are kept. If even the best chunk scores below a threshold, the question is refused.
3. **Answer.** The chunks and the question go to `llama3.2:3b` through Ollama (temperature 0). The reply is streamed to the page.
4. **Quiz.** The model writes a question from one chunk. Python then rejects vague questions, formulas, duplicate or circular options, and any question whose evidence sentence is not word-for-word in the chunk. A second model call answers the question blind, and a question is dropped if the answers differ.

Backend: FastAPI. Frontend: React with Vite.

## Run it

Needs Python 3.11 or newer, Node.js and [Ollama](https://ollama.com). Developed on Windows with Python 3.14.

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

Open http://localhost:5173 and upload a PDF. The API documentation is at http://127.0.0.1:8000/docs.

## Tests and evaluation

```
pip install -r requirements-dev.txt
pytest                   # 41 offline tests
python eval.py           # retrieval benchmark
python eval_answers.py   # end-to-end answer benchmark
```

The benchmark uses three documents with different layouts (a 42-page research paper, a 9-slide lecture deck and a 3-page textbook chapter), with 31 retrieval questions and 27 answer questions. I wrote and rated them myself, so it is a development benchmark, not a general result. The documents are not included in the repository.

| Measure | Result |
|---|---|
| Evidence chunk in the top 6, embeddings only | 81 % |
| Evidence chunk in the top 6, with the reranker | 87 % |
| Good answers (read by hand) | 23 of 27 (2 partial, 2 wrong) |
| Numbers in an answer that are not in the retrieved text | 0 of 27 |
| Off-topic questions refused | 3 of 3 |
| Quiz questions after filtering (10 rated by hand) | 6 good, 4 weak, 0 flawed |

The same settings scored 25 good answers on an earlier run, so differences of one to three answers are within the noise.

Three things the benchmark taught me:

- My first retrieval metric counted a hit whenever any chunk from the right page came back. Checking that the chunk contains the answer text lowered the paper's top-6 score from 94 % to 75 %.
- Larger chunks and keyword search improved retrieval but gave worse answers from the 3B model, which does better with a short context. I chose the settings by answer quality.
- The refusal threshold is set from data: the lowest in-scope score was -6.3 and the highest off-topic score was -10.2, so it sits at -8.

## Limitations

- A few questions still fail, for example "which datasets are used" on the paper, because the chunk listing them is not retrieved.
- The 3B model sometimes misreads tables or adds details that are not in the text.
- Typeset equations come out as symbols and are hard to retrieve.
- Only English PDFs with a text layer. There is no OCR, so scanned files are rejected.
- Single user, no authentication. Not meant to be exposed to the internet.

## Layout

```
rag.py             ingestion, retrieval, prompts, command line
quiz.py            quiz generation and its filters
main.py            FastAPI app
eval.py            retrieval benchmark
eval_answers.py    answer benchmark
diagnose.py        shows what is stored in the database
tests/             offline tests
frontend/          React app
```
