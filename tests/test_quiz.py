"""Quiz generation: the rules that keep bad questions out (pure Python, no models)."""
import json

import quiz

FILLER = " This filler sentence only makes the chunk long enough and is not part of the evidence." * 3


def make(question, correct, wrong, evidence, text=None):
    llm = lambda prompt, t: json.dumps(
        {"question": question, "correct_answer": correct, "wrong_answers": wrong, "evidence": evidence}
    )
    return quiz.make_question(text or (evidence + FILLER), llm)


def test_vague_questions_are_rejected():
    assert quiz.is_vague("What is the method being tested in the experiment?")
    assert quiz.is_vague("What is the method described in Figure 13?")
    assert quiz.is_vague("What is the paper about?")
    assert quiz.is_vague("Which one?")  # too short
    assert not quiz.is_vague("Why does SpectralCanary place the trace in low frequencies?")
    assert not quiz.is_vague("How do I configure the verifier?")  # 'configure' contains 'figure'


def test_evidence_must_be_copied_word_for_word():
    chunk = "The main experiments use LoRA, a common parameter-efficient form of T2I adaptation. More text."
    ok = "The main experiments use LoRA, a common parameter-efficient form of T2I adaptation."
    assert quiz.evidence_ok(ok, "LoRA", chunk)
    assert quiz.evidence_ok(ok.replace(" ", "\n"), "LoRA", chunk)  # line breaks do not matter
    assert not quiz.evidence_ok("The main experiments use full fine-tuning of every layer.", "LoRA", chunk)
    assert not quiz.evidence_ok(ok, "Transfer Learning", chunk)  # the answer must appear in the evidence
    assert not quiz.evidence_ok("uses LoRA", "LoRA", chunk)  # too short to prove anything


def test_numbers_in_the_answer_must_be_in_the_evidence():
    chunk = "After five epochs, AUC reaches 0.956 and TPR at 1% FPR reaches 0.730; performance saturates later."
    evidence = "After five epochs, AUC reaches 0.956 and TPR at 1% FPR reaches 0.730;"
    assert quiz.evidence_ok(evidence, "0.956", chunk)
    assert not quiz.evidence_ok(evidence, "0.965", chunk)


def test_circular_answers_are_rejected():
    assert quiz.is_circular("What type of samples are injected in EnTruth?", "Injected Samples")
    assert not quiz.is_circular("What is the value of Kcal used for SD1.5?", "1000")


def test_near_duplicate_options_are_rejected():
    a = "Blends a fixed canary through a smooth low-pass mask M with strength alpha, retaining most of the spectrum."
    b = "Blends a fixed canary through a smooth low-pass mask M with strength alpha and retains most of the spectrum"
    assert quiz.has_near_duplicates([a, b, "Retains the original canary", "Adds noise to captions only"])
    assert not quiz.has_near_duplicates(["100", "1000", "500", "2000"])  # short numbers are not compared


def test_percentage_questions_need_a_percentage():
    assert quiz.unit_mismatch("What percentage of generations are used?", "Each checkpoint uses 100 generations.")
    assert not quiz.unit_mismatch("What percentage of requests complete?", "95% of requests complete in two seconds.")


def test_formulas_are_detected():
    assert quiz.math_count("𝑋0(𝑓) = 𝜇(𝑓)") >= 4
    assert quiz.math_count("What is alpha = 0.02 in the paper's “quote” – café?") == 0


def test_a_good_question_is_kept_and_options_are_shuffled():
    positions = set()
    for _ in range(40):
        q = make("What is the goal of proactive traceability?", "to preserve customization utility while making later use auditable",
                 ["to prevent customization", "to modify a collection", "to test a trained model"],
                 "Proactive traceability aims to preserve customization utility while making later use auditable.")
        assert q and len(set(q["options"])) == 4 and q["options"][q["answer"]].startswith("to preserve")
        positions.add(q["answer"])
    assert len(positions) > 1  # the right answer is not always in the same place


def test_bad_llm_output_is_skipped():
    assert quiz.make_question("text " * 100, lambda p, t: "not json") is None
    assert make("What is the goal here today?", "A", ["A", "b"], "x" * 50) is None  # fewer than 3 distinct wrong answers


def test_agreement_check_drops_questions_the_checker_disagrees_with():
    q = {"question": "Q?", "options": ["a", "b", "c", "d"], "answer": 2}
    assert quiz.agrees(q, "text", lambda p, t: json.dumps({"answer": " c "}))
    assert not quiz.agrees(q, "text", lambda p, t: json.dumps({"answer": "A"}))
    assert not quiz.agrees(q, "text", lambda p, t: "garbage")


def test_generate_quiz_spreads_pages_and_gives_up():
    text = ("Low frequencies encode broad spatial structure that neural networks preferentially learn and that is important "
            "for generation. A smooth low pass mask weights the frequencies and a small blending strength controls the change. ") * 2
    chunks = [(text + str(i), page) for i, page in enumerate([2, 2, 4, 4, 9, 9, 10])]
    counter = {"n": 0}

    def llm(prompt, t):
        if prompt.startswith("Read the text"):
            line = [l for l in prompt.splitlines() if l[1:3] == ") " and l.endswith("blending strength")][0]
            return json.dumps({"answer": line[0]})
        counter["n"] += 1
        ev = "A smooth low pass mask weights the frequencies and a small blending strength controls the change."
        return json.dumps({"question": f"Question number {counter['n']} about the blending?", "correct_answer": "blending strength",
                           "wrong_answers": ["w1", "w2", "w3"], "evidence": ev})

    questions = quiz.generate_quiz(chunks, 3, llm)
    assert len(questions) == 3 and len({q["page"] for q in questions}) == 3
    assert quiz.generate_quiz(chunks, 3, lambda p, t: "garbage") == []  # never loops forever
