
import json
import random
import re
from difflib import SequenceMatcher

PROMPT = (
    "Write ONE multiple-choice question that tests understanding of the text below.\n"
    "Rules:\n"
    "- Ask about one specific fact, number, definition or reason stated in the text.\n"
    "- The question must make sense on its own: name the method, dataset or experiment it is about. "
    "Never write 'the text', 'the passage', 'the experiment' or 'the paper', and never refer to a figure, table or algorithm number.\n"
    "- Do not ask generic 'what method' or 'what dataset' questions.\n"
    "- The correct answer must be fully supported by the text and add nothing the text does not say.\n"
    "- The correct answer must not repeat the words of the question.\n"
    "- The 3 wrong answers must be plausible but clearly wrong according to the text, and must differ in meaning "
    "(not just in wording) from the correct answer and from each other.\n"
    "- 'evidence' must be ONE sentence copied word for word from the text that proves the correct answer.\n"
    "Reply with JSON only, in exactly this shape: "
    '{{"question": "...", "correct_answer": "...", "wrong_answers": ["...", "...", "..."], "evidence": "..."}}\n\n'
    "Text:\n{text}"
)

CHECK_PROMPT = (
    "Read the text and answer the multiple-choice question using only the text. "
    'Reply with JSON only, for example {{"answer": "B"}} (the letter of the single best option).\n\n'
    "Text:\n{text}\n\nQuestion: {question}\n{options}"
)

VAGUE_PHRASES = ("the text", "the passage", "the experiment", "the paper", "the study",
                 "described in", "this method", "the method being", "the following")
VAGUE_WORDS = re.compile(r"\b(figure|fig\.|table|tab\.|algorithm|appendix|appx\.)", re.IGNORECASE)


MATH_CHARS = re.compile("[\U0001D400-\U0001D7FF\u2211\u222B\u2208\u2264\u2265\u2299\u2016\u2225\u2248\u223C\u2192\u2282\u222A\u2229\u221A\u221E"
                        "\u03B1-\u03C9\u0394\u03A3\u03A9\u0398\u039B\u03A0\u03A6\u03A8]")


def math_count(s):
    """How many math symbols (italic math letters, Greek letters, operators) a string contains."""
    return len(MATH_CHARS.findall(s))


def content_words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4 or w.isdigit()}


def is_circular(question, correct):
    """The answer just repeats the question (for example 'injected samples' for 'what samples are injected?')."""
    words = content_words(correct)
    return bool(words) and words <= content_words(question)


def has_near_duplicates(options):
    """Two long options that say the same thing in different words make the question unfair."""
    norm = [normalize(re.sub(r"[^\w\s]", " ", o)) for o in options]
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            if len(norm[i]) >= 20 and len(norm[j]) >= 20 and SequenceMatcher(None, norm[i], norm[j]).ratio() > 0.85:
                return True
    return False


def unit_mismatch(question, evidence):
    """A question about a 'percentage' needs a percentage in the evidence (not a plain count)."""
    return "percent" in question.lower() and "%" not in evidence and "percent" not in evidence.lower()


def is_quiz_chunk(text):
    """Skip short chunks, reference lists and chunks that are mostly symbols or numbers."""
    refs = sum(text.count(w) for w in ("http", "arXiv", "Proceedings", "et al"))
    letters = sum(c.isalpha() or c == " " for c in text)
    return len(text) >= 300 and refs < 2 and letters / len(text) > 0.8


def is_vague(question):
    q = question.lower()
    return len(q.split()) < 6 or any(p in q for p in VAGUE_PHRASES) or bool(VAGUE_WORDS.search(q))


def normalize(s):
    return " ".join(s.lower().split())


def evidence_ok(evidence, correct, text):
    """The evidence must appear word for word in the chunk and must contain the correct answer's key words."""
    ev = normalize(evidence)
    if len(ev) < 30 or ev not in normalize(text):
        return False
    words = [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", correct.lower()) if len(w) >= 3 or w.isdigit()]
    if not words:
        return False
    found = sum(1 for w in words if w in ev)
    return found / len(words) >= 0.6


def make_question(text, llm):
    """Ask the LLM for one question about this text. Returns a dict, or None if the output is unusable."""
    try:
        data = json.loads(llm(PROMPT.format(text=text), 0.4))
        question = str(data["question"]).strip()
        correct = str(data["correct_answer"]).strip()
        evidence = str(data["evidence"]).strip()
        wrong = []
        for w in data["wrong_answers"]:
            w = str(w).strip()
            if w and w.lower() != correct.lower() and w.lower() not in [x.lower() for x in wrong]:
                wrong.append(w)
    except (ValueError, KeyError, TypeError):
        return None

    if not question or not correct or len(wrong) < 3:
        return None
    if is_vague(question) or not evidence_ok(evidence, correct, text):
        return None
    options = [correct] + wrong[:3]
    if any(math_count(s) >= 2 for s in [question] + options) or math_count(evidence) >= 4:  # formulas make poor multiple-choice items
        return None
    if is_circular(question, correct) or has_near_duplicates(options) or unit_mismatch(question, evidence):
        return None
    random.shuffle(options)  # the right answer should not always be first
    return {
        "question": question,
        "options": options,
        "answer": options.index(correct),  # index of the correct option
        "explanation": f"\u201c{evidence}\u201d",
    }


def agrees(q, text, llm):
    """A second LLM call answers the question from the text without seeing the key.
    If it picks a different option, the question is ambiguous or wrong, so we drop it."""
    options = "\n".join(f"{'ABCD'[i]}) {o}" for i, o in enumerate(q["options"]))
    try:
        data = json.loads(llm(CHECK_PROMPT.format(text=text, question=q["question"], options=options), 0))
        return str(data["answer"]).strip().upper()[:1] == "ABCD"[q["answer"]]
    except (ValueError, KeyError, TypeError):
        return False


def generate_quiz(chunks, n, llm):
    """chunks: list of (text, page). Returns up to n questions, spread over different pages."""
    pool = [(t, p) for t, p in chunks if is_quiz_chunk(t)]
    random.shuffle(pool)

    # one chunk per page first, so the quiz covers the whole document
    seen, first, rest = set(), [], []
    for t, p in pool:
        (rest if p in seen else first).append((t, p))
        seen.add(p)

    quiz, asked = [], set()
    for text, page in (first + rest)[: 4 * n]:  # give up after 4n tries so it cannot run forever
        if len(quiz) == n:
            break
        q = make_question(text, llm)
        if q is None or q["question"].lower() in asked or not agrees(q, text, llm):
            continue
        asked.add(q["question"].lower())
        q["page"] = page
        quiz.append(q)
    return quiz
