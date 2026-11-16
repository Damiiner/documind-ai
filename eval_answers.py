
import re
import sys
import time
import unicodedata

import rag

warning = rag.check_index()
if warning:
    raise SystemExit("STOP - " + warning)

PAPER = "CStheory.pdf"
SLIDES = "intro_to_cybersecurity_lecture_slides.pdf"
CHAPTER = "software_engineering_fundamentals_chapter.pdf"

# (question, required facts, file, pages that contain the answer)
# each fact is a list of accepted spellings; the answer must contain at least one of them
TESTS = [
    # --- research paper
    ("What is SpectralCanary's average per-generation TPR at 1% FPR across the eight Pokemon settings?",
     [["91.1"]], PAPER, {1, 3, 6, 7, 33, 34}),
    ("How long does SpectralCanary take to protect one image on CPU?",
     [["0.39"]], PAPER, {3, 8, 19, 34}),
    ("What default values of alpha, c and delta does SpectralCanary use?",
     [["0.02"], ["0.20", "0.2 ", "0.2,", "0.2.", "0.2)"]], PAPER, {6, 16, 18, 21}),
    ("Which models are fine-tuned in the main experiments and with what method?",
     [["sd1.5", "stable diffusion v1.5", "stable diffusion 1.5"], ["lora"]], PAPER, {6, 16, 19}),
    ("What verifier architecture does SpectralCanary use?",
     [["resnet-50", "resnet50", "resnet 50"]], PAPER, {16}),
    ("How does the defender decide whether a suspect model was trained on the protected data?",
     [["kolmogorov", "ks test", "ks-based", "ks multi", "threshold"]], PAPER, {5, 6, 15, 20}),
    ("Which baseline methods is SpectralCanary compared against?",
     [["siren"], ["diagnosis"], ["entruth"]], PAPER, {3, 6}),
    ("What happens when an attacker tries to remove the trace?",
     [["25", "fid", "quality", "degrad", "damage"]], PAPER, {1, 3, 9, 10}),
    ("Why does SpectralCanary place the trace in the low-frequency part of the spectrum?",
     [["preferentially", "broad"]], PAPER, {2, 3, 4, 9}),
    ("What do the authors say about using only one fixed canary image?",
     [["key reuse", "independently chosen", "broader evaluation"]], PAPER, {10}),
    ("Which datasets are used in the experiments?",
     [["pokemon"], ["rocov2"], ["rsicd"]], PAPER, {6}),
    ("How many generations are used to check whether a suspect model used the protected data?",
     [["30"]], PAPER, {3, 6, 15, 20}),
    # --- lecture slides
    ("What do confidentiality, integrity and availability mean?",
     [["authorized", "authorised"], ["accura"], ["accessib", "when needed", "available when"]], SLIDES, {3}),
    ("How is risk defined in security analysis?",
     [["likelihood", "probab", "chance"], ["impact", "consequence"]], SLIDES, {4, 9}),
    ("What is a denial of service attack?",
     [["overwhelm", "overload", "flood"]], SLIDES, {5}),
    ("What is the principle of least privilege?",
     [["only the access", "access it needs", "needs", "necessary", "minimum"]], SLIDES, {6}),
    ("What are the steps of incident response?",
     [["prepar"], ["identif"], ["contain"], ["recover", "eradicat"], ["learn", "lesson"]], SLIDES, {7}),
    ("What should an employee do after getting an alert about a login from an unfamiliar location?",
     [["official", "portal", "verify"], ["revoke", "reset", "mfa"]], SLIDES, {8}),
    ("What is the difference between a threat and a vulnerability?",
     [["weak"], ["harm", "damage"]], SLIDES, {4}),
    # --- textbook chapter
    ("How does a functional requirement differ from a non-functional requirement?",
     [["behavior", "behaviour", "capabilit"], ["quality", "constraint"]], CHAPTER, {1}),
    ("What do high cohesion and low coupling mean?",
     [["responsibilit"], ["depend"]], CHAPTER, {1, 2}),
    ("What is a regression test?",
     [["previously"]], CHAPTER, {2, 3}),
    ("What are the four kinds of software maintenance?",
     [["corrective"], ["adaptive"], ["perfective"], ["preventive"]], CHAPTER, {2}),
    ("What is an example of a boundary test for an age field?",
     [["minimum"]], CHAPTER, {2}),
    ("What is technical debt?",
     [["shortcut", "future cost", "workaround", "later work"]], CHAPTER, {2, 3}),
    ("How can the vague requirement 'the application should be fast' be made testable?",
     [["95%"], ["two seconds", "2 seconds"]], CHAPTER, {1}),
    ("Why can testing not prove that a program has no defects?",
     [["finite"]], CHAPTER, {2}),
]

# questions that are NOT about the documents: the system must refuse them
OUT_OF_SCOPE = [
    "Who won the 2018 World Cup?",
    "What is the capital of Tunisia?",
    "How do I bake sourdough bread?",
]


def plain(text):
    """Lowercase and remove accents, so 'Pokémon' matches 'pokemon'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def numbers(text):
    """Numbers in an answer, ignoring list markers such as '1.' at the start of a line or '2)' in a sentence."""
    text = re.sub(r"(?m)^\s*\d{1,2}\.\s", " ", text)
    text = re.sub(r"(?<![\w.])\(?\d{1,2}\)\s", " ", text)
    return re.findall(r"\d+(?:\.\d+)?", text)


def run_question(question, facts, filename, pages):
    result = rag.answer_question(question)
    answer = plain(result["answer"])
    context = " ".join(doc for doc, meta, score in rag.retrieve(question)).lower()

    missing = [f[0] for f in facts if not any(alt in answer for alt in f)]
    pages_ok = any(s["source"] == filename and s["page"] in pages for s in result["sources"])
    unsupported = sorted({n for n in numbers(answer) if n not in context})
    return {"answer": result["answer"], "missing": missing, "pages_ok": pages_ok, "unsupported": unsupported}


arg = sys.argv[1] if len(sys.argv) > 1 else ""
if arg.isdigit():
    tests = TESTS[: int(arg)]
elif arg:
    tests = [t for t in TESTS if arg.lower() in t[2].lower()]
else:
    tests = TESTS
if not tests:
    sys.exit(f"No questions match '{arg}'")

rows = []
start = time.time()
log = open("answers_log.txt", "w", encoding="utf-8")  # read this file: keyword checks can be wrong in both directions
for i, (question, facts, filename, pages) in enumerate(tests, start=1):
    r = run_question(question, facts, filename, pages)
    rows.append(r)
    log.write(f"Q: {question}\nA: {r['answer']}\nExpected: {filename} pages {sorted(pages)}\n\n")
    log.flush()
    flags = []
    if r["missing"]:
        flags.append(f"missing facts {r['missing']}")
    if not r["pages_ok"]:
        flags.append("cited source/page wrong")
    if r["unsupported"]:
        flags.append(f"numbers not in context {r['unsupported']}")
    print(f"[{i}/{len(tests)}] {'PASS' if not flags else 'FAIL'} | {question}", flush=True)
    if flags:
        print("         " + "; ".join(flags))
        print("         answer: " + r["answer"].replace("\n", " ")[:300])

n = len(rows)
facts_ok = sum(1 for r in rows if not r["missing"])
pages_ok = sum(1 for r in rows if r["pages_ok"])
clean = sum(1 for r in rows if not r["unsupported"])
all_ok = sum(1 for r in rows if not r["missing"] and r["pages_ok"] and not r["unsupported"])

print("\nOut-of-scope questions (must be refused):")
refused = 0
for question in OUT_OF_SCOPE:
    answer = rag.answer_question(question)["answer"]
    ok = "could not find" in answer.lower()
    refused += ok
    print(f"  {'refused' if ok else 'NOT refused'} | {question}")
    if not ok:
        print("         answer: " + answer.replace("\n", " ")[:200])

print(f"\nrequired facts present : {facts_ok}/{n}")
print(f"cited source+page right: {pages_ok}/{n}")
print(f"no unsupported numbers : {clean}/{n}")
print(f"everything passed      : {all_ok}/{n}")
print(f"out-of-scope refused   : {refused}/{len(OUT_OF_SCOPE)}")
print(f"time: {(time.time() - start) / 60:.1f} min")
print("All answers were saved to answers_log.txt - read a few, including the PASS ones.")
