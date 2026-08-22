"""Deterministic query derivation and rg-based file ranking for the
code-grounding baseline.

The baseline models "what can plain rg/Git retrieve from an issue narrative
alone". Every step is a pure function of the fixture case text so the same
fixture revision always produces the same ranking.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

WEIGHT_EXACT = 3.0
WEIGHT_IDENTIFIER = 2.0
WEIGHT_PROSE = 1.0
PATH_BONUS = 1.0

MAX_EXACT_QUERIES = 8
MAX_IDENTIFIERS = 10
MAX_PROSE_TERMS = 6

_STOPWORDS = frozenset("""a an and are as at be been being but by can cannot could did do does doing
done during each for from had has have having he her here hers him his how i if
in into is it its itself just like may me might more most must my no nor not of
off on once only or other our ours out over own same she should so some such
than that the their theirs them then there these they this those through to too
under until up very was we were what when where which while who whom why will
with without would you your yours it's don't doesn't didn't won't can't""".split())

_WORD_RE = re.compile(r"[a-z]+")
# snake_case, camelCase, SCREAMING_CASE identifiers with a digit or a second
# hump so ordinary english words never qualify.
_IDENT_RE = re.compile(
    r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"
    r"|[A-Z][A-Z0-9]{3,})\b"
)
_ERROR_LINE_RE = re.compile(r"Traceback|^[A-Za-z_.]*Error\b|\berror:|\bException\b")


@dataclass
class Queries:
    exact: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    prose: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "exact": list(self.exact),
            "identifiers": list(self.identifiers),
            "prose": list(self.prose),
        }


def derive_queries(title: str, body: str) -> Queries:
    """Derive the fixed baseline query set from an issue narrative."""
    text = f"{title}\n{body}"
    exact: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = value.strip()
        if 3 <= len(value) <= 160 and value not in seen:
            seen.add(value)
            exact.append(value)

    for span in re.findall(r"`([^`\n]{3,160})`", body):
        add(span)
    for span in re.findall(r'"([^"\n]{3,160})"', body):
        add(span)

    for line in body.splitlines():
        stripped = line.strip()
        if stripped and _ERROR_LINE_RE.search(stripped) and not stripped.startswith("-"):
            add(stripped[:120])

    identifiers: list[str] = []
    ident_seen: set[str] = set()
    for match in _IDENT_RE.findall(text):
        low = match.lower()
        if low in _STOPWORDS:
            continue
        if match not in ident_seen:
            ident_seen.add(match)
            identifiers.append(match)
    ranked = sorted(identifiers, key=lambda i: (-_IDENT_RE.findall(text).count(i), i))

    title_words = [w for w in _WORD_RE.findall(title.lower()) if w not in _STOPWORDS]
    deduped_prose: list[str] = []
    for word in title_words:
        if word not in deduped_prose:
            deduped_prose.append(word)

    return Queries(
        exact=exact[:MAX_EXACT_QUERIES],
        identifiers=ranked[:MAX_IDENTIFIERS],
        prose=deduped_prose[:MAX_PROSE_TERMS],
    )


def run_rg(pattern: str, tree: Path, fixed: bool = True) -> dict[str, int]:
    """Run one rg query over *tree* with default filters and return per-file counts.

    Default rg behaviour is intentional: hidden files, gitignored paths and
    binary files stay invisible because measuring that blindness is part of
    the benchmark (skipped-file rate / fallback escape rate).
    """
    cmd = ["rg", "--no-messages", "-c"]
    if fixed:
        cmd.append("-F")
    else:
        cmd.append("-e")
    cmd.extend([pattern, str(tree)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        path, _, count = line.rpartition(":")
        if path and count.isdigit():
            rel = str(Path(path).resolve().relative_to(tree.resolve()))
            counts[rel] = counts.get(rel, 0) + int(count)
    return counts


def rank_files(queries: Queries, tree: Path) -> dict[str, float]:
    """Score every file that answers at least one derived query."""
    scores: dict[str, float] = {}

    def bump(rel: str, weight: float, count: int) -> None:
        scores[rel] = scores.get(rel, 0.0) + weight * (1.0 + math.log2(max(count, 1)))

    all_terms: list[tuple[str, float]] = [(q, WEIGHT_EXACT) for q in queries.exact] + [
        (q, WEIGHT_IDENTIFIER) for q in queries.identifiers
    ]
    for term, weight in all_terms:
        counts = run_rg(term, tree, fixed=True)
        for rel, count in counts.items():
            bump(rel, weight, count)
    for term in queries.prose:
        counts = run_rg(re.escape(term), tree, fixed=False)
        for rel, count in counts.items():
            bump(rel, WEIGHT_PROSE, count)

    lowered_terms = [t.lower() for t in queries.exact + queries.identifiers + queries.prose]
    for rel in list(scores):
        rel_low = rel.lower()
        if any(term in rel_low for term in lowered_terms if len(term) >= 4):
            scores[rel] += PATH_BONUS
    return scores


def top_k(scores: dict[str, float], k: int) -> list[str]:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [rel for rel, _ in ordered[:k]]


def git_grep_fallback(pattern: str, tree: Path) -> list[str]:
    """Exhaustive fallback: index-based search that sees tracked files rg's
    default filters would hide (hidden paths, tracked-but-gitignored paths).
    Binary files stay excluded via -I, matching the direct-lane contract."""
    proc = subprocess.run(
        ["git", "grep", "-I", "-l", "-F", pattern, "--", "."],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tree),
    )
    return [p[2:] if p.startswith("./") else p for p in proc.stdout.splitlines()]


def file_is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return True
    return b"\x00" in chunk


def skip_reason(tree: Path, rel: str) -> str | None:
    """Why default-filter rg cannot see *rel*: 'ignored', 'hidden', 'binary',
    or None when visible."""
    parts = Path(rel).parts
    if any(part.startswith(".") for part in parts[:-1]) or parts[-1].startswith("."):
        # .gitignore-style directories such as .husky hold real fix targets;
        # rg hides them unless --hidden is passed, which the baseline does not.
        return "hidden"
    proc = subprocess.run(
        # --no-index mirrors rg traversal semantics: a tracked file that
        # matches .gitignore is still invisible to default-filter rg.
        ["git", "check-ignore", "-q", "--no-index", rel],
        cwd=str(tree),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return "ignored"
    if file_is_binary(tree / rel):
        return "binary"
    return None
