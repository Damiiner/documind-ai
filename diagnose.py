
import statistics
import unicodedata

import rag


def plain(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return " ".join(text.split())


data = rag.collection.get(include=["documents", "metadatas"])
docs, metas = data["documents"], data["metadatas"]
if not docs:
    raise SystemExit("The database is empty. Ingest the PDFs first.")

lens = [len(d) for d in docs]
print(f"CHUNK_SIZE in rag.py: {rag.CHUNK_SIZE}")
print("index check:", rag.check_index() or "OK, the database matches rag.py")
print(f"{len(docs)} chunks stored | length min {min(lens)} / average {int(statistics.mean(lens))} / max {max(lens)}")
print("   for comparison: chunk size 900 gives about 156 chunks (average near 720), chunk size 500 gives about 301 (average near 424)")
per_file = {}
for m in metas:
    per_file[m["source"]] = per_file.get(m["source"], 0) + 1
for name, n in sorted(per_file.items()):
    print(f"   {name}: {n} chunks")

# the two questions whose evidence chunk was never found
CHECKS = [
    ("datasets question", "CStheory.pdf", "Pokemon-BLIP-Captions", ["BLIP", "Sketch-Scene", "Datasets and models", "RSICD"]),
    ("baselines question", "CStheory.pdf", "We compare with SIREN", ["We compare", "SIREN", "Comparisons and stress"]),
]
for name, filename, exact, parts in CHECKS:
    mine = [(d, m) for d, m in zip(docs, metas) if m["source"] == filename]
    pages = sorted(m["page"] for d, m in mine if plain(exact) in plain(d))
    print(f"\n{name}: the evidence text '{exact}' is in {len(pages)} stored chunk(s)" + (f", on pages {pages}" if pages else " (NOT FOUND)"))
    for part in parts:
        found = [(m["page"], d) for d, m in mine if plain(part) in plain(d)]
        print(f"   '{part}': {len(found)} chunk(s), pages {sorted({p for p, d in found})[:10]}")
        if found:
            page, d = found[0]
            i = plain(d).find(plain(part))
            print(f"        e.g. p.{page}: ...{plain(d)[max(0, i - 60): i + 110]}...")
