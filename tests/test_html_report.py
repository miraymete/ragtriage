import json
import re
import textwrap
from html.parser import HTMLParser

import numpy as np
import pytest
from typer.testing import CliRunner

from ragtriage import cli
from ragtriage.config import load_config
from ragtriage.datasets import resolve_quote
from ragtriage.evaluate import ExampleScore
from ragtriage.generation.generator import GenerationRecord, GenerationStatus, Generator
from ragtriage.generation.judge import FaithfulnessJudge, FaithfulnessResult
from ragtriage.generation.llm import LLMError, LLMErrorKind, StopReason
from ragtriage.generation.relevance import RelevanceJudge, RelevanceResult
from ragtriage.generation.testing import FakeLLM, fake_completion
from ragtriage.html_report import (
    HtmlDetail,
    ReportData,
    ReportOptions,
    _origin,
    highlight,
    question_issues,
    render_report,
)
from ragtriage.metrics.retrieval import Failure
from ragtriage.metrics.statistics import (
    BootstrapSettings,
    PermutationSettings,
    compare_experiments,
    statistic_metrics,
)
from ragtriage.models import Document, Evidence
from ragtriage.outcomes import Failed, NotApplicable, Scored
from ragtriage.runner import run_retrieval

runner = CliRunner()

FULL = ReportOptions(detail=HtmlDetail.FULL)

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


def make_config(tmp_path, chunk_size=30, extra="", k=1):
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
            k: {k}
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
    page = render_report(report_data(make_config(tmp_path)), FULL)
    check_page(page)
    assert "<h2>Retrieval</h2>" in page
    assert "<h2>Question-level uncertainty</h2>" not in page
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
    assert "<h2>Question-level uncertainty</h2>" in page
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
    args = ["run", str(path), "--html", str(html_path), "--html-detail", "full"]
    result = runner.invoke(cli.app, args)
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
        ReportData(**{**data.__dict__, "experiments": experiments}),  # type: ignore[arg-type]
        FULL,
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


def _run_pipeline(tmp_path, generator_llm, judge_llm=None, chunk_size=200, k=1):
    import asyncio

    from ragtriage.generation.pipeline import generate_reports

    data = report_data(make_config(tmp_path, chunk_size=chunk_size, k=k))
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


# --- compact report and display depth ---------------------------------------------------------

QUESTIONS_HEADING = "<section><h2>Questions</h2>"
BLOCK = re.compile('<(?:details|div) class="q">')


def _record(**update):
    fields = {
        "example_id": "q",
        "answerable": True,
        "status": GenerationStatus.OK,
        "abstained": False,
        "answer": "Liman.",
        "contexts": (),
        "evidence_in_context": True,
        "provider": "fake",
        "model": "m",
        "prompt_version": "v",
        "latency_s": 0.0,
        "abstention": Scored(value=1.0),
    }
    return GenerationRecord(**{**fields, **update})


def _score(failure):
    return ExampleScore(example_id="q", scores={"mrr": 1.0}, failure=failure)


def _faithfulness(outcome):
    return {"faithfulness": FaithfulnessResult(outcome=outcome)}


def _relevance(outcome):
    return {"relevance": RelevanceResult(outcome=outcome)}


@pytest.mark.parametrize(
    "update",
    [
        {},
        {"abstained": True, "answer": "", **_faithfulness(NotApplicable(reason="abstained"))},
        {"abstained": True, "answer": "", **_relevance(NotApplicable(reason="abstained"))},
        _faithfulness(Scored(value=1.0)),
        _relevance(Scored(value=-0.9)),  # relevance has no threshold
    ],
    ids=["ok", "abstained", "relevance_abstained", "faithful", "low_rel"],
)
def test_expected_results_are_not_problems(update):
    assert question_issues(_score(Failure.OK), _record(**update)) == ()


@pytest.mark.parametrize(
    ("update", "kind", "label"),
    [
        ({"abstention": Scored(value=0.0)}, "generation", "abstained on answerable"),
        (
            {"answerable": False, "abstention": Scored(value=0.0)},
            "generation",
            "answered unanswerable",
        ),
        (_faithfulness(Scored(value=0.5)), "generation", "unsupported claims"),
        (_faithfulness(Failed(reason="bad")), "judge", "faithfulness failed"),
        (_relevance(Failed(reason="bad")), "judge", "relevance failed"),
        (
            _faithfulness(NotApplicable(reason="uncertain_claims")),
            "judge",
            "faithfulness not evaluated: uncertain_claims",
        ),
        (
            _faithfulness(NotApplicable(reason="no_claims")),
            "judge",
            "faithfulness not evaluated: no_claims",
        ),
        (
            _relevance(NotApplicable(reason="empty_answer")),
            "judge",
            "relevance not evaluated: empty_answer",
        ),
        (
            _relevance(NotApplicable(reason="new_reason")),
            "judge",
            "relevance not evaluated: new_reason",
        ),
    ],
    ids=[
        "abstained",
        "answered",
        "unsupported",
        "faith",
        "rel",
        "uncertain",
        "no_claims",
        "empty",
        "unknown",
    ],
)
def test_real_problems_are_selected(update, kind, label):
    [issue] = question_issues(_score(Failure.OK), _record(**update))
    assert (issue.kind, issue.label) == (kind, label)


def test_a_failed_generation_is_one_problem_not_also_its_expected_na():
    record = _record(
        status=GenerationStatus.INVALID_OUTPUT,
        abstained=None,
        answer=None,
        abstention=Failed(reason="invalid_output"),
        **_faithfulness(NotApplicable(reason="no_valid_answer")),
    )
    [issue] = question_issues(_score(Failure.OK), record)
    assert issue.label == "generation invalid_output"


@pytest.mark.parametrize("failure", [Failure.RANKING, Failure.RETRIEVAL])
def test_retrieval_failures_are_selected(failure):
    [issue] = question_issues(_score(failure), None)
    assert issue.kind == failure.value


def test_an_unanswerable_question_without_generation_is_not_a_problem():
    assert question_issues(None, None) == ()


def _with_failures(data, failures):
    """Force the failure types, so selection does not depend on what BM25 happened to return."""
    experiments = []
    for result, runs in data.experiments:
        scores = tuple(
            s.model_copy(update={"failure": failures.get(s.example_id, Failure.OK)})
            for s in result.report.per_example
        )
        report = result.report.model_copy(update={"per_example": scores})
        experiments.append((result.model_copy(update={"report": report}), runs))
    return ReportData(**{**data.__dict__, "experiments": experiments})


def test_compact_writes_passages_only_for_selected_questions(tmp_path):
    data = _with_failures(
        report_data(make_config(tmp_path, chunk_size=200)), {"q1": Failure.RANKING}
    )
    page = render_report(data)
    check_page(page)
    questions = page.split(QUESTIONS_HEADING)[1]
    # One block per experiment for q1. q2 and the unanswerable q3 are rows, without a passage.
    assert questions.count('<details class="q">') == 2
    assert questions.count("<td>q2</td>") == 2 and questions.count("<td>q3</td>") == 2
    assert "Without retrieval passages (1)" in questions
    assert "Labeled unanswerable (1)" in questions
    assert "<mark>İzmir bir liman" in questions
    assert "İzmir &amp; &quot;liman&quot; &lt;b&gt;nasıl&lt;/b&gt; bir kent?" in questions
    assert "small &lt;i&gt;" in questions and "small <i>" not in questions
    # The rule and the counts are on the page: 3 questions, 1 with passages, 2 summary only, then
    # retrieval 0, ranking 1, generation 0, judge 0.
    assert "Compact report." in page and "--html-detail full" in page
    assert "<td>3</td><td>1</td><td>2</td><td>0</td><td>1</td><td>0</td><td>0</td>" in page
    for (result, runs), _ in zip(data.experiments, range(2), strict=False):
        written = sum(len(r.chunks) for r in runs if r.example_id == "q1")
        omitted = sum(len(r.chunks) for r in runs if r.example_id != "q1")
        assert f"<td>{written}</td><td>0</td><td>{omitted}</td>" in page, result.name


def test_full_detail_writes_every_question(tmp_path):
    data = report_data(make_config(tmp_path, chunk_size=200))
    page = render_report(data, FULL)
    check_page(page)
    assert page.count('<details class="q">') == 6
    assert "Without retrieval passages (" not in page
    assert "Full detail." in page


def test_summary_sections_and_json_do_not_depend_on_the_html_options(tmp_path):
    path = make_config(tmp_path)
    common = ["run", str(path), "--stats", "--resamples", "200"]
    common += ["--baseline", "small <i>", "--candidate", "large"]
    pages, results = [], []
    for name, options in (
        ("compact", []),
        ("full", ["--html-detail", "full"]),
        ("depth", ["--html-detail", "full", "--html-depth", "1"]),
    ):
        html_path, out = tmp_path / f"{name}.html", tmp_path / f"{name}.json"
        args = [*common, "--html", str(html_path), "--out", str(out), *options]
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 0, result.output
        pages.append(html_path.read_text("utf-8").split(QUESTIONS_HEADING)[0])
        loaded = json.loads(out.read_text("utf-8"))
        for item in loaded:  # wall-clock timings differ between any two runs
            del item["index_seconds"], item["retrieval_ms"]
        results.append(loaded)
    assert "Recall@1" in pages[0] and "Paired comparison" in pages[0]
    assert pages[0] == pages[1] == pages[2]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("detail", [HtmlDetail.COMPACT, HtmlDetail.FULL])
def test_display_depth_hides_ranks_but_never_the_generation_context(tmp_path, detail):
    # Invalid answers select every question, so both modes write the blocks.
    script = [fake_completion("not json")] * 6
    data = report_data(make_config(tmp_path, k=2))
    generator = Generator(FakeLLM(script), k=data.k, min_overlap=data.min_overlap)
    import asyncio

    from ragtriage.generation.pipeline import generate_reports

    runs = [list(r) for _, r in data.experiments]
    reports = asyncio.run(generate_reports(generator, data.examples, runs))
    full_data = ReportData(**{**data.__dict__, "generation": reports})
    page = render_report(full_data, ReportOptions(detail=detail, max_rank=1))
    check_page(page)
    first_run = next(r for r in data.experiments[0][1] if r.example_id == "q1")
    assert len(first_run.chunks) >= 2
    blocks = BLOCK.split(page)[1:]
    block = next(b for b in blocks if "<strong>q1</strong>" in b)
    retrieved, context = block.split("Context actually sent to the model (2 chunks)")
    assert "rank 1 ·" in retrieved and "rank 2 ·" not in retrieved
    assert "rank 1 ·" in context and "rank 2 ·" in context
    hidden = len(first_run.chunks) - 1
    assert f"{hidden} more retrieved results not written here" in retrieved
    assert f"(ranks 2–{len(first_run.chunks)})" in retrieved
    assert "the display depth is 1" in retrieved
    assert "Metrics still" in retrieved
    assert "never cut by this limit" in page
    total_hidden = sum(len(r.chunks) - min(len(r.chunks), 1) for r in data.experiments[0][1])
    total_written = sum(min(len(r.chunks), 1) for r in data.experiments[0][1])
    assert f"<td>{total_written}</td><td>{total_hidden}</td><td>0</td>" in page


def test_no_hidden_note_when_the_limit_changes_nothing(tmp_path):
    data = report_data(make_config(tmp_path, chunk_size=200))
    assert max(len(r.chunks) for _, runs in data.experiments for r in runs) <= 5
    page = render_report(data, ReportOptions(detail=HtmlDetail.FULL, max_rank=5))
    assert "not written here" not in page
    assert "Display depth: at most 5" in page


def _judged_reports(tmp_path, n_experiments=2):
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
    generator = _stub([answer(), answer(), answer(True, "")] * n_experiments)
    judge = _stub([claims, verdicts] * 2 * n_experiments)
    return _run_pipeline(tmp_path, generator, judge)


def _block_of(section, question_id):
    return next(b for b in BLOCK.split(section)[1:] if f"<strong>{question_id}</strong>" in b)


def test_compact_keeps_answers_outcomes_and_context_for_every_question(tmp_path):
    data, reports = _judged_reports(tmp_path)
    page = render_report(ReportData(**{**data.__dict__, "generation": reports}))
    check_page(page)
    questions = page.split(QUESTIONS_HEADING)[1].split("<strong>large</strong>")[1]
    # Nothing here has a problem, so gold evidence and retrieved lists are really left out.
    assert "<h4>Gold evidence</h4>" not in questions and "Top 1 (what" not in questions
    assert '<details class="q">' not in questions
    for question_id in ("q1", "q2"):
        block = _block_of(questions, question_id)
        visible = block.split("<details>")[0]
        assert "Liman &lt;script&gt;x&lt;/script&gt; &amp; &quot;kent&quot;." in visible
        assert "answered" in visible and "faithfulness" in visible and "scored 1.000" in visible
        assert "Context actually sent to the model (1 chunks)" in block
        assert "<strong>c1</strong>" in block  # judge details stay too
    abstained = _block_of(questions, "q3")
    visible = abstained.split("<details>")[0]
    assert "No answer text: the model abstained." in visible
    assert ">abstained<" in visible and "N/A: abstained" in visible
    n_sent = len(next(r for r in reports[1].records if r.example_id == "q3").contexts)
    assert f"Context actually sent to the model ({n_sent} chunks)" in abstained
    assert "Model abstained" in page and "1 (faithfulness: abstained 1)" in page


def test_context_sent_is_complete_for_every_question_under_a_display_depth(tmp_path):
    generator = _stub([answer(), answer(), answer(True, "")] * 2)
    data, reports = _run_pipeline(tmp_path, generator, chunk_size=30, k=2)
    options = ReportOptions(max_rank=1)
    page = render_report(ReportData(**{**data.__dict__, "generation": reports}), options)
    check_page(page)
    sections = page.split(QUESTIONS_HEADING)[1].split("<strong>large</strong>")
    assert any(len(r.contexts) == 2 for report in reports for r in report.records)
    for section, report in zip(sections, reports, strict=True):
        for record in report.records:
            block = _block_of(section, record.example_id)
            assert f"model ({len(record.contexts)} chunks)" in block
            context = block.split("Context sent to the model")[-1]
            context = context.split("Context actually sent to the model")[-1]
            for span in record.contexts:
                assert f"rank {span.rank} · {span.doc_id} [{span.start}, {span.end})" in context


def _with_outcomes(reports, by_id):
    records = [
        r.model_copy(update=by_id[r.example_id]) if r.example_id in by_id else r
        for r in reports[0].records
    ]
    return [reports[0].model_copy(update={"records": tuple(records)}), reports[1]]


def test_abstention_failed_generation_and_not_evaluated_are_shown_apart(tmp_path):
    data, reports = _judged_reports(tmp_path)
    changed = _with_outcomes(
        reports,
        {
            "q1": {
                "status": GenerationStatus.INVALID_OUTPUT,
                "abstained": None,
                "answer": None,
                "problem": "bozuk çıktı",
                "abstention": Failed(reason="invalid_output"),
                **_faithfulness(NotApplicable(reason="no_valid_answer")),
            },
            "q2": _faithfulness(NotApplicable(reason="no_claims")),
        },
    )
    page = render_report(ReportData(**{**data.__dict__, "generation": changed}))
    check_page(page)
    first = page.split(QUESTIONS_HEADING)[1].split("<strong>large</strong>")[0]
    failed, no_claims, abstained = (_block_of(first, q) for q in ("q1", "q2", "q3"))
    assert "not run: no_valid_answer" in failed and "No valid answer: bozuk çıktı" in failed
    assert "not evaluated: no_claims" in no_claims and "badge warn" in no_claims
    assert "N/A: abstained" in abstained and "not evaluated" not in abstained
    assert "N/A: no_claims" not in page and "N/A: no_valid_answer" not in page
    # One row per column, under the first experiment.
    row = page.split("Why a judge gave no score")[1].split("<tr>")[2]
    cells = re.findall("<td>(.*?)</td>", row)
    assert cells[1:] == [
        "1 (faithfulness: abstained 1)",
        "1 (faithfulness: no_valid_answer 1)",
        "1 (faithfulness: no_claims 1)",
    ]
    # q1 and q2 have problems, so their passages are written. q3's abstention is expected.
    assert "<h4>Gold evidence</h4>" in failed and "<h4>Gold evidence</h4>" in no_claims
    assert "<h4>Gold evidence</h4>" not in abstained
    assert "No evidence is labeled" not in abstained
    assert "Top 1 (what" not in abstained


def test_an_unexpected_na_reason_is_selected_and_escaped(tmp_path):
    data, reports = _judged_reports(tmp_path)
    odd = NotApplicable(reason='<b>tuhaf</b> & "ış"')
    changed = _with_outcomes(reports, {"q1": _faithfulness(odd)})
    page = render_report(ReportData(**{**data.__dict__, "generation": changed}))
    check_page(page)
    assert "<b>tuhaf</b>" not in page
    assert "faithfulness not evaluated: &lt;b&gt;tuhaf&lt;/b&gt; &amp; &quot;ış&quot;" in page
    assert page.count("<h4>Gold evidence</h4>") == 1


def test_compact_stop_keeps_the_question_that_stopped_the_run(tmp_path):
    error = LLMError(LLMErrorKind.AUTH, "invalid x-api-key", status_code=401)
    data, reports = _run_pipeline(tmp_path, _stub([fake_completion(OK_ANSWER), error]))
    page = render_report(ReportData(**{**data.__dict__, "generation": reports}))
    check_page(page)
    first = page.split(QUESTIONS_HEADING)[1].split("<strong>large</strong>")[0]
    stopped = _block_of(first, "q2")
    assert "stopped the run here" in stopped
    assert "The run stopped before this question finished" in stopped
    assert "<h4>Gold evidence</h4>" in stopped
    assert page.count("<h4>Gold evidence</h4>") == 1
    # q3 was never reached: it says so, and there is nothing else to show for it.
    assert "no generation result" in _block_of(first, "q3")
    assert "<details>" not in _block_of(first, "q3")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--html-depth", "2"], "need --html"),
        (["--html-detail", "full"], "need --html"),
        (["--html", "r.html", "--html-depth", "0"], "between 1 and the run's depth (5)"),
        (["--html", "r.html", "--html-depth", "6"], "between 1 and the run's depth (5)"),
    ],
    ids=["depth_alone", "detail_alone", "zero", "above_depth"],
)
def test_invalid_html_options_stop_before_any_work(tmp_path, monkeypatch, options, message):
    def fail(*_):
        raise AssertionError("retrieval must not start")

    monkeypatch.setattr(cli, "run_retrieval", fail)
    result = runner.invoke(cli.app, ["run", str(make_config(tmp_path)), *options])
    assert result.exit_code == 2
    assert message in result.output


def test_an_unknown_detail_mode_is_rejected(tmp_path):
    args = ["run", str(make_config(tmp_path)), "--html", "r.html", "--html-detail", "tiny"]
    assert runner.invoke(cli.app, args).exit_code == 2


def test_render_report_checks_the_display_depth_too(tmp_path):
    data = report_data(make_config(tmp_path))
    with pytest.raises(ValueError, match="between 1 and the run's depth"):
        render_report(data, ReportOptions(max_rank=6))
