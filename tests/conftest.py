"""
Shared setup for the tests. They run offline and fast:
- the real embedding and reranker models are replaced by empty stand-ins
- rag.py creates a "db" folder in the current directory when it is imported, so the tests run inside a
  temporary folder and can never touch your real database or uploads
"""
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

class FakeModel:
    """Stands in for both SentenceTransformer and CrossEncoder."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True):
        return types.SimpleNamespace(tolist=lambda: [[0.0, 0.0, 0.0, 1.0] for _ in texts])

    def predict(self, pairs):
        return [0.0 for _ in pairs]


fake = types.ModuleType("sentence_transformers")
fake.SentenceTransformer = FakeModel
fake.CrossEncoder = FakeModel
sys.modules["sentence_transformers"] = fake

os.chdir(tempfile.mkdtemp(prefix="documind_tests_"))
