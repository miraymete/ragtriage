import textwrap
from html.parser import HTMLParser

import numpy as np
import pytest
from typer.testing import CliRunner

from ragtriage import cli
from ragtriage.config import load_config
from ragtriage.datasets import resolve_quote
from ragtriage.generation.generator import Generator
from ragtriage.generation.judge import FaithfulnessJudge
from ragtriage.generation.llm import LLMError, LLMErrorKind, StopReason
from ragtriage.generation.relevance import RelevanceJudge
from ragtriage.generation.testing import FakeLLM, fake_completion
from ragtriage.html_report import ReportData, _origin, highlight, render_report
from ragtriage.metrics.statistics import (
    BootstrapSettings,
    PermutationSettings,
    compare_experiments,
    statistic_metrics,
)
from ragtriage.models import Document, Evidence
from ragtriage.runner import run_retrieval

runner = CliRunner()

DOC = 'Liman & "başlık" <script>alert(1)</script>. İzmir bir liman kentidir. Ankara başkenttir.'
NASTY_QUESTION = 'İzmir & "liman" <b>nasıl</b> bir kent?'


class Balance(HTMLParser):
    """Checks that every non-void tag is closed in order and that no script tag appears."""

    VOID = frozenset({"meta", "br", "hr", "img", "input", "link"})

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts += 1
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.errors.append(tag)


def check_page(page):
    parser = Balance()
    parser.feed(page)
    parser.close()
    assert parser.errors == [], parser.errors
    assert parser.stack == []
    assert parser.scripts == 0
    assert "http://" not in page and "https://" not in page
    # The raw payload from the document never reaches the page unescaped.
    assert "<script>alert(1)</script>" not in page


def make_config(tmp_path, chunk_size=30, extra=""):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "a.txt").write_text(DOC, "utf-8")
    (tmp_path / "qa.jsonl").write_text(
        f'{{"id": "q1", "question": {_json(NASTY_QUESTION)}, '
        '"evidence": [{"doc_id": "a", "quote": "İzmir bir liman kentidir."}]}\n'
        '{"id": "q2", "question": "Ankara ne?", '
        '"evidence": [{"doc_id": "a", "quote": "Ankara başkenttir."}]}\n'
        '{"id": "q3", "question": "Efes nerede?"}\n',
        "utf-8",
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            f"""\
            documents: docs
            dataset: qa.jsonl
            k: 1
            depth: 5
            experiments:
              - name: small <i>
                chunker: {{type: fixed, size: {chunk_size}}}
                retriever: {{type: bm25}}
              - name: large
                chunker: {{type: fixed, size: 200}}
                retriever: {{type: bm25}}
            """
        )
        + extra,
        "utf-8",
    )
    return path


def _json(text):
    import json

    return json.dumps(text, ensure_ascii=False)


def report_data(path, **extra):
    config = load_config(path)
    outputs = run_retrieval(config)
    return ReportData(
        source=str(path),
        k=config.k,
        depth=config.depth,
        min_overlap=config.min_overlap,
        documents=outputs.documents,
        examples=outputs.examples,
        experiments=outputs.experiments,
        **extra,
    )


# --- highlighting -----------------------------------------------------------------------------


def test_highlight_escapes_every_segment_and_marks_the_overlap():
    text = 'ab<&>"ç'
    # The chunk starts at document offset 10; the gold span [12, 14) is "<&".
    evidence = [Evidence(doc_id="d", start=12, end=14)]
    assert highlight(text, "d", 10, evidence) == "ab<mark>&lt;&amp;</mark>&gt;&quot;ç"


def test_highlight_clips_merges_and_ignores_other_documents():
    evidence = [
        Evidence(doc_id="d", start=5, end=11),  # starts before the chunk
        Evidence(doc_id="d", start=12, end=14),
        Evidence(doc_id="d", start=13, end=16),  # overlaps the previous one
        Evidence(doc_id="other", start=10, end=17),
    ]
    assert highlight("abcdefg", "d", 10, evidence) == "<mark>a</mark>b<mark>cdef</mark>g"
    assert highlight("abc", "d", 10, []) == "abc"


def test_highlight_uses_python_character_offsets_for_turkish_text():
    document = Document(id="d", text='Şehir: İzmir & "Ankara" ığüşöç')
    gold = resolve_quote(document, 'İzmir & "Ankara"')
    assert (gold.start, gold.end) == (7, 23)
    page = highlight(document.text, "d", 0, [gold])
    assert page == "Şehir: <mark>İzmir &amp; &quot;Ankara&quot;</mark> ığüşöç"
    # A chunk that starts inside the span keeps the same characters marked.
    assert highlight(document.text[10:], "d", 10, [gold]).startswith("<mark>ir &amp;")


def test_origin_labels_keep_fake_and_cache_apart():
    assert "FAKE" in _origin("fake", None)
    assert "FAKE" in _origin("cache", "fake") and "from cache" in _origin("cache", "fake")
    assert _origin("cache", "api") == (
        '<span class="badge na">cache (originally a real API call)</span>'
    )
    assert "real API call" in _origin("api", None)
    assert "no response" in _origin(None, None)


# --- retrieval only ---------------------------------------------------------------------------


def test_retrieval_only_report(tmp_path):
    page = render_report(report_data(make_config(tmp_path)))
    check_page(page)
    assert "<h2>Retrieval</h2>" in page
    assert "<h2>Uncertainty</h2>" not in page
    assert "<h2>Generation</h2>" not in page
    assert "<h4>Generation" not in page
    assert "small &lt;i&gt;" in page and "small <i>" not in page
    assert "İzmir &amp; &quot;liman&quot; &lt;b&gt;nasıl&lt;/b&gt; bir kent?" in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "plain text" in page
    assert "2 questions scored; 1 labeled unanswerable" in page
    # Gold evidence is shown with its span, and the matching chunk text is marked.
    start = DOC.index("İzmir bir liman kentidir.")
    assert f"a [{start}, {start + 25})" in page
    assert "<mark>İzmir bir liman" in page
    # Top k and deeper results are separate.
    assert "Top 1 (what the generator is given)" in page
    assert "Deeper results, ranks 2–" in page
    assert "results at these ranks were not added to the generation context" in page
    assert "never sent" not in page
    # Unanswerable questions are listed apart and not called failures.
    assert "Labeled unanswerable (1)" in page
    assert "It is not a retrieval failure" in page
    assert "Sorted by recall@1 (lowest first), then reciprocal rank" in page


def test_questions_are_sorted_by_recall_then_reciprocal_rank_then_id(tmp_path):
    data = report_data(make_config(tmp_path))
    page = render_report(data)
    result = data.experiments[0][0]
    by_id = {s.example_id: s.scores for s in result.report.per_example}
    expected = sorted(by_id, key=lambda i: (by_id[i]["recall@1"], by_id[i]["mrr"], i))
    block = page.split("<strong>small &lt;i&gt;</strong>")[1].split("Labeled unanswerable")[0]
    found = [i for i in sorted(by_id, key=block.find)]
    assert found == expected


# --- statistics -------------------------------------------------------------------------------


def test_statistics_section_shows_methods_and_unadjusted_p(tmp_path):
    path = make_config(tmp_path)
    html_path = tmp_path / "report.html"
    args = ["run", str(path), "--stats", "--resamples", "200", "--html", str(html_path)]
    args += ["--baseline", "small <i>", "--candidate", "large"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    page = html_path.read_text("utf-8")
    check_page(page)
    assert "<h2>Uncertainty</h2>" in page
    assert "Percentile bootstrap over questions: 200 resamples, seed 0" in page
    assert "Paired comparison: large − small &lt;i&gt;" in page
    assert "<strong>unadjusted</strong>" in page
    assert "(exact," in page
    assert "different methods" in page
    assert "not adjusted for multiple comparisons" in page
    assert "winner" not in page.replace("No result here names a winner", "")


def test_monte_carlo_comparison_is_labeled(tmp_path):
    data = report_data(make_config(tmp_path))
    (a, _), (b, _) = data.experiments

    def shifted(report, delta):
        # The same questions with lower scores, so every difference is nonzero.
        scores = tuple(
            s.model_copy(update={"scores": {m: v - delta for m, v in s.scores.items()}})
            for s in report.per_example
        )
        return report.model_copy(update={"per_example": scores})

    comparison = compare_experiments(
        (a.name, shifted(a.report, 0.5)),
        (b.name, b.report),
        statistic_metrics(1),
        BootstrapSettings(n_resamples=100),
        PermutationSettings(n_draws=50, max_exact_nonzero=0),
    )
    assert all(m.permutation.method == "monte_carlo" for m in comparison.metrics)
    page = render_report(
        ReportData(**{**data.__dict__, "comparison": comparison})  # type: ignore[arg-type]
    )
    check_page(page)
    assert "Monte Carlo, 50 draws, seed 0, p = (extreme + 1) / (draws + 1)" in page
    assert "(exact," not in page
    assert "Mean with" not in page  # no per-experiment intervals were computed


# --- generation with fake models --------------------------------------------------------------

GENERATION = """\
generation:
  generator: {provider: anthropic, model: gen-model}
"""


def fake_generator(script):
    def build(spec, k, min_overlap):
        return Generator(FakeLLM(script), k=k, min_overlap=min_overlap)

    return build


def answer(abstained=False, text='Liman <script>x</script> & "kent".'):
    return fake_completion({"abstained": abstained, "answer": text})


class Stub:
    def embed_query(self, text):
        return np.array([1.0, 0.0])

    def embed_documents(self, texts):
        raise AssertionError


def test_fake_generation_with_judges(tmp_path, monkeypatch):
    path = make_config(tmp_path, chunk_size=200)
    script = [answer(), answer(), answer(True, "")] * 2
    monkeypatch.setattr(cli, "build_generator", fake_generator(script))
    claims = fake_completion({"claims": ["İzmir bir liman kentidir."]})
    verdicts = fake_completion(
        {
            "verdicts": [
                {
                    "claim_id": "c1",
                    "verdict": "supported",
                    "passage_id": "p1",
                    "quote": "liman kentidir",
                }
            ]
        }
    )
    monkeypatch.setattr(
        cli,
        "build_judge",
        lambda spec: FaithfulnessJudge(FakeLLM([claims, verdicts] * 4, model="judge")),
    )
    questions = fake_completion({"questions": ["Bir & <i>?", "İki?", "Üç?"]})
    monkeypatch.setattr(
        cli,
        "build_relevance",
        lambda spec: RelevanceJudge(FakeLLM([questions] * 4), Stub(), {"type": "stub <x>"}),
    )
    config = GENERATION + "  judge: {provider: anthropic, model: judge}\n"
    config += "  relevance:\n    judge: {provider: anthropic, model: rel}\n"
    config += "    embedder: {type: hashing}\n"
    path.write_text(path.read_text("utf-8") + config, "utf-8")
    html_path = tmp_path / "report.html"
    result = runner.invoke(cli.app, ["run", str(path), "--html", str(html_path)])
    assert result.exit_code == 0, result.output
    page = html_path.read_text("utf-8")
    check_page(page)
    assert "Liman &lt;script&gt;x&lt;/script&gt; &amp; &quot;kent&quot;." in page
    assert "scripted fake model" in page and "FAKE: scripted test model" in page
    assert "each counted once: 0 (0 returned a response" in page
    assert "Not included: 0 cache hits and 18 scripted fake calls" in page
    assert "not HTTP request counts" in page
    assert "not validated against human labels" in page
    assert "<h3>Faithfulness</h3>" in page and "<h3>Answer relevance</h3>" in page
    assert "N/A: abstained" in page
    assert "Bir &amp; &lt;i&gt;? · cosine 1.000" in page
    assert "stub &lt;x&gt;" in page
    assert "Context actually sent to the model (1 chunks)" in page
    assert "one: generator" not in page  # roles are named after the experiment
    assert "large: generator" in page and "large: faithfulness judge" in page
    assert "large: relevance judge" in page
    assert "stopped" not in page.lower().replace("stopped early", "")


def test_failed_generation_and_partial_run(tmp_path, monkeypatch):
    path = make_config(tmp_path, chunk_size=200)
    path.write_text(path.read_text("utf-8") + GENERATION, "utf-8")
    script = [
        fake_completion("not json <b>"),
        LLMError(LLMErrorKind.AUTH, 'no credentials & "key" <missing>'),
    ]
    monkeypatch.setattr(cli, "build_generator", fake_generator(script))
    html_path = tmp_path / "report.html"
    result = runner.invoke(cli.app, ["run", str(path), "--html", str(html_path)])
    assert result.exit_code == 2
    page = html_path.read_text("utf-8")
    check_page(page)
    # The first question failed and is shown as a failed generation, with escaped raw output.
    assert "failed: invalid_output" in page
    assert "not json &lt;b&gt;" in page
    # The run then stopped: the reason is escaped, the rest has no result.
    assert "Stopped in small &lt;i&gt;:</strong> auth: no credentials &amp; &quot;key&quot;" in page
    assert "1 of 3 questions have a result" in page
    assert "The run stopped before this question finished" in page
    assert "no generation result" in page
    # The second experiment never started generation.
    assert "Generation was not started for this experiment" in page
    assert "not started" in page
    assert "Wrote" in result.output


def test_report_needs_an_experiment(tmp_path):
    data = report_data(make_config(tmp_path))
    with pytest.raises(ValueError, match="no experiments"):
        render_report(ReportData(**{**data.__dict__, "experiments": []}))  # type: ignore[arg-type]


def test_a_deeper_duplicate_of_a_top_k_chunk_is_explained(tmp_path):
    data = report_data(make_config(tmp_path))
    result, runs = data.experiments[0]
    run = next(r for r in runs if r.chunks)
    # The top chunk comes back again at rank 2, below k = 1.
    doubled = run.model_copy(update={"chunks": (run.chunks[0], run.chunks[0])})
    experiments = [(result, [doubled if r is run else r for r in runs]), *data.experiments[1:]]
    page = render_report(
        ReportData(**{**data.__dict__, "experiments": experiments})  # type: ignore[arg-type]
    )
    check_page(page)
    assert "duplicate of rank 1, ignored by metrics" in page
    assert "same chunk is in the generation context at rank 1" in page


def test_call_attempts_exclude_cache_and_fake_and_count_failures(tmp_path, monkeypatch):
    path = make_config(tmp_path, chunk_size=200)
    path.write_text(path.read_text("utf-8") + GENERATION, "utf-8")
    ok = {"abstained": False, "answer": "Liman."}
    script = [
        fake_completion(ok, source="api"),
        fake_completion("not json", source="api"),  # invalid output, still a response
        LLMError(LLMErrorKind.SERVER, "boom", status_code=500),  # no response
        fake_completion(ok, source="cache", original_source="api"),
        fake_completion(ok, stop_reason=StopReason.REFUSAL, source="api"),
        fake_completion(ok, stop_reason=StopReason.MAX_TOKENS, source="api"),
    ]

    def build(spec, k, min_overlap):
        llm = FakeLLM(script)
        llm.provider = "stub-provider"  # stands in for a real provider
        return Generator(llm, k=k, min_overlap=min_overlap)

    monkeypatch.setattr(cli, "build_generator", build)
    html_path = tmp_path / "report.html"
    result = runner.invoke(cli.app, ["run", str(path), "--html", str(html_path)])
    assert result.exit_code == 0, result.output
    page = html_path.read_text("utf-8")
    check_page(page)
    assert "each counted once: 5 (4 returned a response" in page
    assert "1 ended in an API error without one" in page
    assert "Not included: 1 cache hits and 0 scripted fake calls" in page
    assert "may also have made the call that stopped it" not in page


# --- the call that stopped the run -------------------------------------------------------------


def _stub(script, provider="stub-provider"):
    llm = FakeLLM(script)
    llm.provider = provider  # stands in for a real provider unless it is "fake"
    return llm


def _run_pipeline(tmp_path, generator_llm, judge_llm=None):
    import asyncio

    from ragtriage.generation.pipeline import generate_reports

    data = report_data(make_config(tmp_path, chunk_size=200))
    generator = Generator(generator_llm, k=data.k, min_overlap=data.min_overlap)
    judge = FaithfulnessJudge(judge_llm) if judge_llm is not None else None
    runs = [list(r) for _, r in data.experiments]
    reports = asyncio.run(generate_reports(generator, data.examples, runs, judge))
    return data, reports


OK_ANSWER = {"abstained": False, "answer": "Liman."}


def test_a_sent_call_that_stops_the_run_is_counted_with_unknown_usage(tmp_path):
    script = [
        fake_completion(OK_ANSWER, source="api"),
        LLMError(LLMErrorKind.AUTH, "invalid x-api-key", status_code=401),
    ]
    data, reports = _run_pipeline(tmp_path, _stub(script))
    [report] = reports
    stop = report.stop_attempt
    assert stop is not None
    assert (stop.role, stop.example_id, stop.provider) == ("generator", "q2", "stub-provider")
    assert (stop.sent_to_api, stop.status_code, stop.usage, stop.cost_usd) == (
        True,
        401,
        None,
        None,
    )
    # The attempt is metadata, not a record: q2 still has no result.
    assert [r.example_id for r in report.records] == ["q1"]
    page = render_report(ReportData(**{**data.__dict__, "generation": reports}))  # type: ignore[arg-type]
    check_page(page)
    assert "each counted once: 2 (1 returned a response" in page
    assert "1 ended in an API error without one" in page
    assert "include 1 call that stopped the run" in page
    assert "tokens and cost are unknown" in page
    assert "generator, call that stopped the run (q2)" in page
    assert "not HTTP request counts" in page


@pytest.mark.parametrize(
    ("provider", "error"),
    [
        # Raised by the SDK before anything is sent: no HTTP status.
        ("stub-provider", LLMError(LLMErrorKind.AUTH, "no Anthropic credentials found")),
        # A scripted fake error, even with a status code, is not a real request.
        ("fake", LLMError(LLMErrorKind.AUTH, "scripted", status_code=401)),
    ],
    ids=["before_sending", "fake_client"],
)
def test_stops_without_a_real_request_are_not_counted(tmp_path, provider, error):
    data, reports = _run_pipeline(tmp_path, _stub([error], provider))
    stop = reports[0].stop_attempt
    assert stop is not None and stop.sent_to_api is False
    page = render_report(ReportData(**{**data.__dict__, "generation": reports}))  # type: ignore[arg-type]
    assert "each counted once: 0 (0 returned a response" in page
    assert "The error that stopped the run was not counted" in page
    assert "call that stopped the run (" not in page


def test_a_cache_error_records_no_stop_attempt(tmp_path):
    from ragtriage.generation.cache import CacheError

    def broken(request):
        raise CacheError("unreadable cache file")

    data, reports = _run_pipeline(tmp_path, _stub(broken))
    [report] = reports
    assert report.stopped == "cache error: unreadable cache file"
    assert report.stop_attempt is None
    page = render_report(ReportData(**{**data.__dict__, "generation": reports}))  # type: ignore[arg-type]
    assert "each counted once: 0 (0 returned a response" in page
    assert "stopped the run" not in page.split("Logical API calls")[1].split("</p>")[0]


def test_a_judge_call_that_stops_the_run_is_attributed_to_the_judge(tmp_path):
    generator = _stub([fake_completion(OK_ANSWER, source="api")])
    judge = _stub([LLMError(LLMErrorKind.PERMISSION, "no access", status_code=403)])
    judge.model = "judge-model"
    _, reports = _run_pipeline(tmp_path, generator, judge)
    stop = reports[0].stop_attempt
    assert stop is not None
    assert (stop.role, stop.model, stop.sent_to_api) == ("faithfulness_judge", "judge-model", True)
    # The answer is kept, and the unfinished judging is marked failed, not scored.
    [record] = reports[0].records
    assert record.answer == "Liman."
    assert record.faithfulness.outcome.status == "failed"
