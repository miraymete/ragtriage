"""Convert SQuAD v1.1-format data (such as XQuAD) into ragtriage documents and examples.

Each paragraph becomes one Document with its text unchanged, so `answer_start` is already a valid
character offset (a Python string index) into that document. Nothing is normalized and no offset
is repaired: an answer whose offset does not point at its text is reported and left out.

SQuAD lists several answers for some questions. They are alternatives written by different
annotators, not separate pieces of evidence that are all required, so only the first answer
becomes Evidence. The others are kept as metadata and never enter a metric.

An answer span is a short phrase. Retrieving it is not the same as retrieving enough context to
answer the question, so results on this kind of label say less than they seem to.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ragtriage.models import Document, Evidence, QAExample


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QuestionMeta(_Frozen):
    """What a QAExample cannot hold: where the question came from, and the other answers."""

    id: str
    article: str
    doc_id: str
    # Further annotator answers that point at their text. Not used by any metric.
    alternatives: tuple[Evidence, ...] = ()


class Issue(_Frozen):
    question_id: str | None
    article: str
    reason: str
    detail: str = ""


class Conversion(_Frozen):
    documents: dict[str, Document]
    examples: tuple[QAExample, ...]
    meta: tuple[QuestionMeta, ...]
    # Records left out, or parts of records dropped, with the reason.
    issues: tuple[Issue, ...]
    # Questions whose answer text occurs more than once in its paragraph. The offset decides
    # which occurrence is meant; a quote alone would be ambiguous.
    n_repeated_answer_text: int
    article_of_document: dict[str, str]


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def document_id(title: str, paragraph_index: int) -> str:
    """A file-name-safe document id. The original title is kept in `article_of_document`."""
    return f"{_UNSAFE.sub('_', title)}__p{paragraph_index:02d}"


def _answer_span(context: str, doc_id: str, answer: Any) -> tuple[Evidence | None, str | None, str]:
    """Return (evidence, problem, detail). Exactly one of evidence and problem is set."""
    if not isinstance(answer, Mapping):
        return None, "malformed_answer", repr(answer)[:80]
    text, start = answer.get("text"), answer.get("answer_start")
    if not isinstance(text, str) or not text:
        return None, "empty_answer", repr(text)[:80]
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        return None, "bad_answer_start", repr(start)[:80]
    found = context[start : start + len(text)]
    if found != text:
        return None, "offset_mismatch", f"expected {text!r}, found {found!r} at {start}"
    return Evidence(doc_id=doc_id, start=start, end=start + len(text)), None, ""


def convert_squad(raw: Mapping[str, Any]) -> Conversion:
    """Convert parsed SQuAD JSON. Structural errors raise; bad records are reported."""
    articles = raw.get("data")
    if not isinstance(articles, list):
        raise ValueError("expected a top-level 'data' list")
    documents: dict[str, Document] = {}
    article_of: dict[str, str] = {}
    examples: list[QAExample] = []
    meta: list[QuestionMeta] = []
    issues: list[Issue] = []
    seen_ids: set[str] = set()
    repeated = 0
    titles: set[str] = set()
    for article in articles:
        title = article.get("title") if isinstance(article, Mapping) else None
        if not isinstance(title, str) or not title:
            raise ValueError(f"article without a title: {repr(article)[:80]}")
        if title in titles:
            raise ValueError(f"duplicate article title {title!r}")
        titles.add(title)
        for index, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context")
            if not isinstance(context, str) or not context:
                raise ValueError(f"{title}: paragraph {index} has no context")
            doc_id = document_id(title, index)
            if doc_id in documents:
                raise ValueError(f"document id collision: {doc_id!r}")
            documents[doc_id] = Document(id=doc_id, text=context)
            article_of[doc_id] = title
            for qa in paragraph.get("qas", []):
                qid, question = qa.get("id"), qa.get("question")
                if not isinstance(qid, str) or not qid:
                    issues.append(Issue(question_id=None, article=title, reason="missing_id"))
                    continue
                if qid in seen_ids:
                    issues.append(Issue(question_id=qid, article=title, reason="duplicate_id"))
                    continue
                seen_ids.add(qid)
                if not isinstance(question, str) or not question.strip():
                    issues.append(Issue(question_id=qid, article=title, reason="empty_question"))
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    # SQuAD v1.1 has no unanswerable questions, so an empty list is a broken
                    # record, not an "unanswerable" label.
                    issues.append(Issue(question_id=qid, article=title, reason="no_answer"))
                    continue
                evidence, problem, detail = _answer_span(context, doc_id, answers[0])
                if evidence is None:
                    issues.append(
                        Issue(question_id=qid, article=title, reason=problem or "", detail=detail)
                    )
                    continue
                alternatives = []
                for other in answers[1:]:
                    alt, alt_problem, alt_detail = _answer_span(context, doc_id, other)
                    if alt is None:
                        issues.append(
                            Issue(
                                question_id=qid,
                                article=title,
                                reason=f"alternative_{alt_problem}",
                                detail=alt_detail,
                            )
                        )
                    elif alt != evidence and alt not in alternatives:
                        alternatives.append(alt)
                answer_text = context[evidence.start : evidence.end]
                if context.count(answer_text) > 1:
                    repeated += 1
                examples.append(
                    QAExample(
                        id=qid,
                        question=question,
                        evidence=(evidence,),
                        reference_answer=answer_text,
                        group=title,
                    )
                )
                meta.append(
                    QuestionMeta(
                        id=qid, article=title, doc_id=doc_id, alternatives=tuple(alternatives)
                    )
                )
    return Conversion(
        documents=documents,
        examples=tuple(examples),
        meta=tuple(meta),
        issues=tuple(issues),
        n_repeated_answer_text=repeated,
        article_of_document=article_of,
    )


# --- splits -----------------------------------------------------------------------------------

Split = Literal["dev", "test"]


def _rank(seed: str, salt: str, title: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{title}".encode()).hexdigest()


def split_articles(titles: Iterable[str], seed: str, n_dev: int) -> dict[str, Split]:
    """Assign whole articles to dev or test, so no article's questions or paragraphs straddle both.

    The order is a hash of the seed and the title: the same inputs always give the same split,
    and the order of the input does not matter.
    """
    ordered = sorted(set(titles), key=lambda t: _rank(seed, "split", t))
    if not 0 < n_dev < len(ordered):
        raise ValueError(f"n_dev must be between 1 and {len(ordered) - 1}, got {n_dev}")
    return {title: ("dev" if i < n_dev else "test") for i, title in enumerate(ordered)}


def pilot_articles(dev_titles: Iterable[str], seed: str, n: int) -> list[str]:
    """A deterministic subset of dev articles for a small pilot."""
    ordered = sorted(set(dev_titles), key=lambda t: _rank(seed, "pilot", t))
    if not 0 < n <= len(ordered):
        raise ValueError(f"n must be between 1 and {len(ordered)}, got {n}")
    return sorted(ordered[:n])


def _normalize(text: str) -> str:
    """For duplicate detection only; never applied to stored text."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class LeakageReport(_Frozen):
    articles_in_both: tuple[str, ...]
    questions_in_both: tuple[str, ...]
    # Same text (after case and whitespace folding) on both sides of the split.
    duplicate_paragraph_texts: tuple[tuple[str, str], ...]
    duplicate_question_texts: tuple[tuple[str, str], ...]

    @property
    def clean(self) -> bool:
        return not (self.articles_in_both or self.questions_in_both)


def check_leakage(conversion: Conversion, split: Mapping[str, Split]) -> LeakageReport:
    """Check that no article or question id is on both sides, and list duplicate texts.

    Each paragraph belongs to one article, so an article-level split keeps paragraphs apart.

    Duplicate texts are reported, not fixed: whether a repeated question across articles is a
    leak depends on what it asks.
    """
    sides: dict[str, set[str]] = {"dev": set(), "test": set()}
    for title, side in split.items():
        sides[side].add(title)
    articles_in_both = tuple(sorted(sides["dev"] & sides["test"]))
    doc_side = {d: split[a] for d, a in conversion.article_of_document.items()}
    question_sides: dict[str, set[str]] = {}
    for m in conversion.meta:
        question_sides.setdefault(m.id, set()).add(split[m.article])
    by_paragraph: dict[str, list[tuple[str, str]]] = {}
    for doc_id, document in conversion.documents.items():
        by_paragraph.setdefault(_normalize(document.text), []).append((doc_id, doc_side[doc_id]))
    question_text = {e.id: e.question for e in conversion.examples}
    by_question: dict[str, list[tuple[str, str]]] = {}
    for m in conversion.meta:
        by_question.setdefault(_normalize(question_text[m.id]), []).append((m.id, split[m.article]))

    def crossing(groups: Mapping[str, list[tuple[str, str]]]) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for items in groups.values():
            dev = sorted(i for i, s in items if s == "dev")
            test = sorted(i for i, s in items if s == "test")
            pairs.extend((d, t) for d in dev for t in test)
        return tuple(sorted(pairs))

    return LeakageReport(
        articles_in_both=articles_in_both,
        questions_in_both=tuple(sorted(q for q, s in question_sides.items() if len(s) > 1)),
        duplicate_paragraph_texts=crossing(by_paragraph),
        duplicate_question_texts=crossing(by_question),
    )


def issue_counts(issues: Sequence[Issue]) -> dict[str, int]:
    return dict(sorted(Counter(i.reason for i in issues).items()))
