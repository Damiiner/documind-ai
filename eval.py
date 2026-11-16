import unicodedata

import rag

warning = rag.check_index()
if warning:
    raise SystemExit("STOP - " + warning)

PAPER = "CStheory.pdf"
SLIDES = "intro_to_cybersecurity_lecture_slides.pdf"
CHAPTER = "software_engineering_fundamentals_chapter.pdf"

# (question, file, evidence snippets: a chunk is a hit if it contains ANY of them)
TESTS = [
    ("What is SpectralCanary's average per-generation TPR at 1% FPR across the eight Pokemon settings?", PAPER, ["91.1%"]),
    ("How long does SpectralCanary take to protect one image on CPU?", PAPER, ["0.39 seconds"]),
    ("What default values of alpha, c and delta does SpectralCanary use?", PAPER, ["Unless stated otherwise, SpectralCanary uses", "operating point"]),
    ("Which models are fine-tuned in the main experiments and with what method?", PAPER, ["main experiments use LoRA"]),
    ("What verifier architecture does SpectralCanary use?", PAPER, ["binary head"]),
    ("How does the defender decide whether a suspect model was trained on the protected data?", PAPER, ["Kolmogorov", "decision threshold"]),
    ("Which baseline methods is SpectralCanary compared against?", PAPER, ["We compare with SIREN"]),
    ("What happens when an attacker tries to remove the trace?", PAPER, ["additional FID points"]),
    ("What is the equation for the protected spectrum?", PAPER, ["The protected spectrum is"]),
    ("What do the authors say about using only one fixed canary image?", PAPER, ["key reuse"]),
    ("How much better is low-frequency placement than high-frequency on ROCOv2?", PAPER, ["88.9 points"]),
    ("Which datasets are used in the experiments?", PAPER, ["Pokemon-BLIP-Captions"]),
    ("Why does SpectralCanary place the trace in the low-frequency part of the spectrum?", PAPER, ["preferentially learn"]),
    ("What limitations do the authors acknowledge?", PAPER, ["does not establish transfer"]),
    ("What four requirements should a practical dataset-use auditing method satisfy?", PAPER, ["four requirements"]),
    ("What happens if an attacker knows the canary and the blending parameters?", PAPER, ["exact inversion"]),
    ("What do confidentiality, integrity and availability mean?", SLIDES, ["visible only to authorized"]),
    ("How is risk defined in security analysis?", SLIDES, ["likelihood and impact"]),
    ("What is a denial of service attack?", SLIDES, ["overwhelming a service"]),
    ("What is the principle of least privilege?", SLIDES, ["Apply least privilege"]),
    ("What are the steps of incident response?", SLIDES, ["Eradicate and recover"]),
    ("What should an employee do after getting an alert about a login from an unfamiliar location?", SLIDES, ["revoke active sessions"]),
    ("What is the difference between a threat and a vulnerability?", SLIDES, ["a weakness that could be exploited"]),
    ("How does a functional requirement differ from a non-functional requirement?", CHAPTER, ["Functional requirements specify behavior"]),
    ("What do high cohesion and low coupling mean?", CHAPTER, ["high cohesion and low coupling", "internal details of other components"]),
    ("What is a regression test?", CHAPTER, ["A regression test captures", "Regression test: a test that checks"]),
    ("What are the four kinds of software maintenance?", CHAPTER, ["Corrective maintenance fixes defects"]),
    ("What is an example of a boundary test for an age field?", CHAPTER, ["accepts an age"]),
    ("What is technical debt?", CHAPTER, ["Technical debt: future cost", "technical debt should be reviewed"]),
    ("How can the vague requirement 'the application should be fast' be made testable?", CHAPTER, ["95% of search requests"]),
    ("Why can testing not prove that a program has no defects?", CHAPTER, ["no finite test suite"]),
]

# questions that are NOT about any of the documents: the system should refuse them
OUT_OF_SCOPE = [
    "Who won the 2018 World Cup?",
    "What is the capital of Tunisia?",
    "How do I bake sourdough bread?",
]


def plain(text):
    """Lowercase, remove accents, collapse whitespace, so snippets match however the PDF spaced the text."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return " ".join(text.split())


def evaluate(question, filename, snippets, rerank, source=None, hybrid=True):
    results = rag.retrieve(question, k=100, rerank=rerank, source=source, hybrid=hybrid)
    if not results:
        return None, float("-inf")
    best = max(score for _, _, score in results)  # best reranker score among the candidates (used by the refusal guard)
    wanted = [plain(s) for s in snippets]
    for i, (doc, meta, score) in enumerate(results, start=1):
        if meta["source"] == filename and any(w in plain(doc) for w in wanted):
            return i, best
    return None, best  # rank of the first chunk that contains the evidence, best score


def fmt(rank):
    return str(rank) if rank else "MISS"


def hit_rates(ranks):
    n = len(ranks)
    parts = []
    for k in (1, 3, 6):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        parts.append(f"hit@{k} {100 * hits / n:3.0f}%")
    return "   ".join(parts)


rows = []
print(f"{'embed':>6} {'rerank':>7} {'hybrid':>7} {'filter':>7} {'top score':>10} | file | question")
for question, filename, snippets in TESTS:
    a, _ = evaluate(question, filename, snippets, rerank=False)
    b, _ = evaluate(question, filename, snippets, rerank=True, hybrid=False)
    c, top = evaluate(question, filename, snippets, rerank=True)
    d, _ = evaluate(question, filename, snippets, rerank=True, source=filename)
    rows.append((filename, a, b, c, d, top))
    print(f"{fmt(a):>6} {fmt(b):>7} {fmt(c):>7} {fmt(d):>7} {top:>10.2f} | {filename[:12]:12} | {question}")

print("\nHit rates (all documents are mixed in the index, except in the 'filter' mode):")
for label, files in (("everything", None), ("paper", [PAPER]), ("slides", [SLIDES]), ("chapter", [CHAPTER])):
    sel = [r for r in rows if files is None or r[0] in files]
    print(f"\n  {label} ({len(sel)} questions)")
    for name, idx in (("embed ", 1), ("rerank", 2), ("hybrid", 3), ("filter", 4)):
        print(f"    {name}: {hit_rates([r[idx] for r in sel])}")

print("\nOut-of-scope questions (best chunk score should be below MIN_SCORE):")
off = []
for question in OUT_OF_SCOPE:
    _, top = evaluate(question, "", set(), rerank=True)
    off.append(top)
    print(f"  {top:>7.2f}  {'refused' if top < rag.MIN_SCORE else 'NOT refused':12}| {question}")
print(f"\nlowest in-scope top score: {min(r[5] for r in rows):.2f}   highest out-of-scope top score: {max(off):.2f}   MIN_SCORE: {rag.MIN_SCORE}")
