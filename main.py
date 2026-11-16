
import json
import os

import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import quiz
import rag

MAX_UPLOAD_MB = 50
MAX_PAGES = 300

app = FastAPI(title="DocuMind AI")
os.makedirs("uploads", exist_ok=True)

# let the React dev server (Vite) call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str  # "user" or "assistant"
    text: str


class Question(BaseModel):
    question: str
    source: str | None = None  # optional: search only inside this document
    history: list[Turn] = Field(default_factory=list, max_length=10)  # earlier turns, for follow-up questions


@app.post("/documents")
def upload_document(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    name = os.path.basename(file.filename)  # never trust a path sent by the client
    path = os.path.join("uploads", name)

    # save the file in 1 MB pieces and stop if it is too big
    size, too_big = 0, False
    with open(path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                too_big = True
                break
            f.write(chunk)
    if too_big:
        os.remove(path)
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")

    # make sure it is a real PDF and not too long.
    # The file is read into memory first so that no handle stays open on it: Windows cannot delete an open file.
    with open(path, "rb") as f:
        data = f.read()
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = len(doc)
    except Exception:
        os.remove(path)
        raise HTTPException(status_code=400, detail="This file is not a valid PDF")
    if pages > MAX_PAGES:
        os.remove(path)
        raise HTTPException(status_code=413, detail=f"PDF has {pages} pages (maximum is {MAX_PAGES})")

    chunks = rag.ingest(path)
    if chunks == 0:
        os.remove(path)
        raise HTTPException(status_code=422, detail="No text found in this PDF (scanned? OCR needed)")
    return {"filename": name, "pages": pages, "chunks": chunks}


@app.get("/documents")
def list_documents():
    return rag.list_documents()


@app.delete("/documents/{name}")
def delete_document(name: str):
    if not rag.delete_document(os.path.basename(name)):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": name}


@app.post("/ask")
def ask_question(body: Question):
    question = rag.standalone_question(body.question, [t.model_dump() for t in body.history])
    result = rag.answer_question(question, source=body.source)
    result["question_used"] = question  # what was really searched (differs for follow-up questions)
    return result


@app.post("/ask_stream")
def ask_stream(body: Question):
    """Same as /ask, but the answer arrives piece by piece as lines of JSON (type: question, sources, token, error)."""
    def lines():
        try:
            question = rag.standalone_question(body.question, [t.model_dump() for t in body.history])
            yield json.dumps({"type": "question", "text": question}) + "\n"
            for event in rag.answer_stream(question, source=body.source):
                yield json.dumps(event) + "\n"
        except Exception as error:  # for example Ollama is not running
            yield json.dumps({"type": "error", "text": f"The answer could not be generated: {error}"}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


class QuizRequest(BaseModel):
    source: str
    n: int = Field(default=5, ge=1, le=10)  # number of questions


@app.post("/quiz")
def make_quiz(body: QuizRequest):
    chunks = rag.get_chunks(body.source)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")
    questions = quiz.generate_quiz(chunks, body.n, llm=lambda p, t: rag.llm(p, json_mode=True, temperature=t, model=rag.QUIZ_MODEL))
    if not questions:
        raise HTTPException(status_code=502, detail="Could not generate questions, please try again")
    return {"source": body.source, "questions": questions}
