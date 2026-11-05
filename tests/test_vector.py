import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ragtriage import Chunk
from ragtriage.pipelines.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from ragtriage.pipelines.vector import VectorRetriever


def make_chunks(*texts):
    return [Chunk(doc_id=f"d{i}", start=0, end=len(t), text=t) for i, t in enumerate(texts)]


class TableEmbedder:
    """Returns chosen vectors, so expected scores can be worked out by hand."""

    def __init__(self, table):
        self.table = table

    def embed_documents(self, texts):
        return np.array([self.table[text] for text in texts], dtype=np.float32)

    def embed_query(self, text):
        return np.array(self.table[text], dtype=np.float32)


def test_vector_retriever_ranks_by_cosine_not_by_magnitude():
    embedder = TableEmbedder(
        {"aligned": [0.1, 0.0], "large": [10.0, 10.0], "opposite": [-1.0, 0.0], "q": [1.0, 0.0]}
    )
    retriever = VectorRetriever(make_chunks("aligned", "large", "opposite"), embedder)
    results = retriever.retrieve("q", top_k=3)
    assert [r.text for r in results] == ["aligned", "large", "opposite"]
    assert [r.score for r in results] == pytest.approx([1.0, 2**-0.5, -1.0])


def test_vector_retriever_respects_top_k_and_breaks_ties_stably():
    embedder = TableEmbedder({"a": [1.0], "b": [1.0], "c": [1.0], "q": [1.0]})
    retriever = VectorRetriever(make_chunks("a", "b", "c"), embedder)
    assert [r.text for r in retriever.retrieve("q", top_k=2)] == ["a", "b"]


def test_zero_vectors_do_not_produce_nan():
    embedder = TableEmbedder(
        {"empty": [0.0, 0.0], "real": [1.0, 0.0], "q": [1.0, 0.0], "z": [0, 0]}
    )
    retriever = VectorRetriever(make_chunks("empty", "real"), embedder)
    results = retriever.retrieve("q", top_k=2)
    assert [r.text for r in results] == ["real", "empty"]
    assert results[1].score == 0.0
    assert retriever.retrieve("z", top_k=2) == []


def test_vector_retriever_rejects_bad_input():
    with pytest.raises(ValueError, match="empty chunk list"):
        VectorRetriever([], TableEmbedder({}))
    embedder = TableEmbedder({"a": [1.0], "q": [1.0]})
    with pytest.raises(ValueError, match="top_k"):
        VectorRetriever(make_chunks("a"), embedder).retrieve("q", top_k=0)


def test_vector_retriever_rejects_wrong_shape():
    class Broken(TableEmbedder):
        def embed_documents(self, texts):
            return np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        VectorRetriever(make_chunks("a", "b"), Broken({}))


def test_hashing_embedder_is_deterministic_and_shaped():
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed_documents(["İznik çinisi", "kahve"])
    second = HashingEmbedder(dim=64).embed_documents(["İznik çinisi", "kahve"])
    assert first.shape == (2, 64)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], embedder.embed_query("İznik çinisi"))


def test_hashing_embedder_empty_input():
    embedder = HashingEmbedder(dim=8)
    assert embedder.embed_documents([]).shape == (0, 8)
    assert not np.any(embedder.embed_query("..."))


def test_hashing_embedder_matches_inflected_turkish_words():
    retriever = VectorRetriever(
        make_chunks("Peribacası adı verilen kaya oluşumları", "Çay fabrikası kuruldu"),
        HashingEmbedder(),
    )
    assert retriever.retrieve("peribacaları", top_k=1)[0].doc_id == "d0"


def test_hashing_embedder_applies_turkish_casing():
    embedder = HashingEmbedder(dim=256)
    assert np.array_equal(embedder.embed_query("KIRMIZI"), embedder.embed_query("kırmızı"))


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"ngram_min": 0}, {"ngram_min": 4, "ngram_max": 3}])
def test_hashing_embedder_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        HashingEmbedder(**kwargs)


def fake_sentence_transformers(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name):
            calls.append(("load", name))

        def encode(self, texts, **kwargs):
            calls.append(("encode", list(texts), kwargs))
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel)
    )
    return calls


def test_sentence_transformer_adapter_applies_the_right_prefix_to_each_side(monkeypatch):
    calls = fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder(
        "some/model", query_prefix="query: ", document_prefix="passage: ", batch_size=7
    )
    documents = embedder.embed_documents(["belge"])
    query = embedder.embed_query("soru")

    assert calls[0] == ("load", "some/model")
    assert calls[1][1] == ["passage: belge"]
    assert calls[1][2]["batch_size"] == 7
    assert calls[1][2]["normalize_embeddings"] is True
    assert calls[2][1] == ["query: soru"]
    assert documents.shape == (1, 3)
    assert query.shape == (3,)


def test_sentence_transformer_adapter_explains_a_missing_install(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"ragtriage\[embeddings\]"):
        SentenceTransformerEmbedder("some/model")
