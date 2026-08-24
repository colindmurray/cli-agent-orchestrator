"""The Muse ``/status`` panel, parsed as pre-task identity evidence.

Muse's managed-v2 readiness is observed from the provider's own ``/status``
panel rather than from a SessionStart hook (Claude) or a minting bootstrap
(Kimi/Codex).  The panel is the provider-owned surface that names the exact
running session, model, reasoning effort, agent profile, provider, cwd, and
pre-task run state — the coordinator no-prompt canary on 2026-08-10 rendered
exactly this panel for a launched ``muse resume <id>`` session (exact
session ``adcb742e-2ab5-4239-9fe2-b503005db341``, agent profile
``native-basic``, provider ``meta``, exact cwd, ``Run: idle``,
``0 tokens / 0 turns``).

The installed 0.1.0-R708.1 meta panel renders model and effort together in
one line::

    Model: muse-spark-1.2-contributor (reasoning high)

There is no separate Reasoning row on that build; the echo provider renders
a bare model with no effort at all.  The parser therefore splits an exact
trailing `` (reasoning <effort>)`` suffix off the Model value into canonical
``model`` and ``reasoning`` fields, and treats a separate
``Reasoning:``/``Reasoning effort:`` row (a separately-supported variant)
as an additional source that must converge with the suffix or be refused as
ambiguous.

The 0.2.1-R1215.1 build replaced that labeled panel with a box drawn in
``┌│└─`` furniture whose rows carry uppercase labels without colons, and
moved the reasoning effort out of the Model value onto a `` · ``-separated
segment (verified live on the installed meta build, where ``/status``
renders ``MODEL  muse-spark-1.2-contributor · ultra`` over the provider and
profile row ``meta · native-basic``, a ``SESSION`` row naming the
canonical-UUID session id, a ``USAGE`` row of ``N tokens · N turns · N
subagents``, and an ``IDLE`` badge on the header row).  Both shapes parse
into the same canonical fields; a capture carrying neither shape falls back
to the persistent inline footer (``<model> · [<effort> ·]
<directory>``), which is route evidence only: it names NO session id and
is returned explicitly marked as a partial observation that no caller may
treat as a ready receipt.

The panel is *printed output*: Muse writes it into the output area and the
composer line stays rendered at the bottom, so no modal dismiss is required
after observation.  That is also why the parse is strict: a panel that is
missing, ambiguous, truncated, or naming anything other than the expected
(or a canonical provider-generated) session is not readiness, and nothing is
admitted on it.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional, Sequence

from cli_agent_orchestrator.services import provider_contracts

#: The exact command typed into the pane to render the status panel.
STATUS_COMMAND = "/status"

#: The agent-profile identity the installed build renders when launched
#: without ``--preset`` — the built-in default preset.  The launch never
#: passes ``--preset`` (only ``native-basic`` and ``miniswe`` exist and
#: neither is a CAO profile), so this is the profile identity the panel
#: must name.  It is a Muse-side fact, distinct from the CAO profile
#: family recorded in the roster and from the profile material digests
#: carried through the launch.
DEFAULT_AGENT_PROFILE = "native-basic"

#: Schema of the parsed, required panel evidence recorded in the
#: bootstrap and the readiness receipt.
STATUS_PANEL_SCHEMA = "cao-muse-status-panel-v1"

#: Panel row labels for a *separate* reasoning row.  The installed
#: 0.1.0-R708.1 meta panel does NOT render one — it puts the effort inside
#: the Model line — but some builds do, and a duplicate source must be
#: handled strictly (identical values converge; conflicting values refuse
#: as ambiguous) rather than ignored.  Both spellings are accepted because
#: a panel that renders the value under either label is the same evidence.
_REASONING_LABELS = ("Reasoning:", "Reasoning effort:")

_REQUIRED_LABELS = (
    "Session:",
    "Model:",
    "Agent profile:",
    "Model provider:",
    "Directory:",
    "Run:",
    "Token usage:",
)

#: Row labels of the 0.2.1+ boxed panel (no colons, uppercase).  The four
#: data rows here are the identity and route evidence; header/access/
#: activity rows are chrome the parser deliberately ignores.
_BOXED_PANEL_LABELS = ("SESSION", "MODEL", "WORKSPACE", "USAGE", "CONTEXT", "ACCESS", "ACTIVITY")

#: The labels every boxed panel must render for the capture to parse:
#: the session id, the model line, its provider/profile continuation row,
#: the workspace directory, and the pre-task usage.
_BOXED_PANEL_REQUIRED = ("SESSION", "MODEL", "WORKSPACE", "USAGE")

#: Furniture characters drawn around both panel generations.  ``┌┐└┘``
#: belong to the 0.2.1+ box; ``╭╰╯`` to the 0.1.x one.
_PANEL_FURNITURE = ("│", "╭", "╰", "┌", "┐", "└", "┘")

#: The exact reasoning-effort vocabulary the installed build accepts for
#: ``--reasoning-effort`` (``muse --help``).  A suffix or separate value
#: outside this set is malformed evidence and is refused, never guessed.
_MUSE_EFFORT_VOCABULARY = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "ultra"})

#: The exact trailing `` (reasoning <effort>)`` suffix the installed meta
#: panel appends to the Model value.  Only this exact form is split off;
#: any other parenthetical remains part of the model value.
_REASONING_SUFFIX = re.compile(r"^(.*?)\s+\(reasoning\s+([^()]*)\)$")

#: Pre-task state required before any durable readiness may be published:
#: the panel's own statement that the session is idle with zero turns.
PRE_TASK_RUN_STATE = "idle"
PRE_TASK_TOKEN_USAGE = (0, 0)


def _split_model_reasoning(value: str) -> tuple[str, Optional[str]]:
    """Split an optional exact `` (reasoning <effort>)`` suffix off a Model.

    Only the exact installed form is split; any other parenthetical is part
    of the model value (never guessed at).  A suffix that looks like the
    form but carries an empty or unknown effort is refused rather than
    guessed, because binding a session on an effort nobody selected is the
    failure this parse exists to prevent.
    """
    value = value.strip()
    match = _REASONING_SUFFIX.fullmatch(value)
    if match is None:
        return value, None
    model, effort = match.group(1).strip(), match.group(2).strip()
    if not effort or effort not in _MUSE_EFFORT_VOCABULARY:
        raise MuseStatusParseError(
            f"the /status Model value carries a malformed reasoning suffix: {value!r}; "
            "refusing rather than guessing an effort from arbitrary parenthetical text"
        )
    return model, effort


def _converge_reasoning(sources: Sequence[Optional[str]]) -> Optional[str]:
    """Converge identical reasoning values, or refuse conflicting ones.

    The model-line suffix and a separate Reasoning row are two sources for
    the same fact.  Identical values agree and converge; conflicting values
    mean the capture cannot prove which effort the session runs, so it is
    refused as ambiguous rather than resolved by a guess.
    """
    present = [source for source in sources if source is not None]
    if not present:
        return None
    first = present[0]
    if any(source != first for source in present[1:]):
        raise MuseStatusParseError(
            "the /status panel reports conflicting reasoning values "
            f"{sorted(set(present))}; refusing rather than guessing which is the "
            "session's effort"
        )
    return first


class MuseStatusParseError(ValueError):
    """The captured screen is not a usable ``/status`` panel."""


class MuseStatusMismatch(ValueError):
    """The panel parsed, but it does not name the claimed pre-task session."""


def _strip_panel_row(row: str) -> str:
    """One composited row with its box-drawing furniture removed.

    Both panel generations render inside boxes drawn with literal
    characters (``╭│╰╯─`` on 0.1.x, ``┌│└┘─`` on 0.2.1+); captures return
    the composited viewport without escape sequences (``capture-pane -p``),
    so the furniture is literal.
    """
    cleaned = row.strip()
    for marker in _PANEL_FURNITURE:
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip("─ ")
    return cleaned.strip()


def _is_boxed_panel(rows: Sequence[str]) -> bool:
    """Whether the capture carries the 0.2.1+ boxed panel shape.

    The box border characters are disjoint from the 0.1.x furniture and at
    least two of the uppercase labels must be present, so a 0.1.x panel,
    a bare footer, or startup banner rows can never be mistaken for this
    shape.
    """
    stripped = [_strip_panel_row(raw) for raw in rows]
    has_box_border = any(("┌" in raw or "└" in raw) for raw in rows)
    labeled = sum(
        1
        for row in stripped
        if re.match(r"^[A-Z][A-Z ]*[A-Z]\s{2,}\S", row) and row.split()[0] in _BOXED_PANEL_LABELS
    )
    return has_box_border and labeled >= 2


def is_recognized_shape(rows: Sequence[str]) -> bool:
    """Whether the capture carries a recognizable ``/status`` rendering.

    Shape detection is deliberately weaker than the strict parse: a box
    the viewport clipped mid-render is still this pane's panel, and
    counting it as unrecognized would fast-fail a healthy launch that is
    one frame away from its evidence.  Callers use this only to keep
    polling on the full runway; nothing is admitted until the strict
    parse succeeds.
    """
    if _is_boxed_panel(rows):
        return True
    return any(
        _strip_panel_row(raw).startswith(_REQUIRED_LABELS + _REASONING_LABELS) for raw in rows
    )


def _split_dot_segments(value: str) -> Optional[list[str]]:
    """Split an exact `` · ``-separated value, or None if it has none."""
    if "·" not in value:
        return None
    segments = [segment.strip() for segment in value.split("·")]
    if any(not segment for segment in segments):
        return None
    return segments


def _parse_boxed_panel(rows: Sequence[str]) -> dict[str, Any]:
    """Parse the 0.2.1+ boxed ``/status`` panel into the canonical fields.

    Rows carry uppercase labels without colons (``SESSION``, ``MODEL``,
    ``WORKSPACE``, ``USAGE``); the model's reasoning effort rides a
    `` · `` segment of the MODEL value rather than a suffix, and the
    provider and agent profile share one continuation row under MODEL.
    Every singleton label must appear exactly once — a second one is
    ambiguity, never a value to pick — and anything missing leaves the
    capture refused rather than guessed at.
    """
    fields: dict[str, list[str]] = {}
    continuations: dict[str, list[str]] = {}
    last_label: Optional[str] = None
    header_run: Optional[str] = None
    for raw in rows:
        row = _strip_panel_row(raw)
        if not row:
            continue
        match = re.match(r"^(?P<label>[A-Z][A-Z ]*[A-Z])\s{2,}(?P<value>\S.*)$", row)
        if match is not None and match.group("label") in _BOXED_PANEL_LABELS:
            label = match.group("label")
            fields.setdefault(label, []).append(match.group("value").strip())
            last_label = label
            continue
        header = re.match(r"^MUSE CODE\s+\S+(?:\s+/\s+\S+)?(?:\s+([A-Za-z]+))?$", row)
        if header is not None and header.group(1):
            header_run = header.group(1).lower()
            continue
        if last_label is not None:
            continuations.setdefault(last_label, []).append(row)

    # Every singleton among the REQUIRED labels is a one-value fact; a
    # second one is ambiguity, never a value to pick.  Chrome labels
    # (CONTEXT, ACCESS, ACTIVITY) are deliberately outside this check: a
    # repeated chrome row says nothing about which session this is.
    duplicates = [label for label in _BOXED_PANEL_REQUIRED if len(fields.get(label, [])) > 1]
    if duplicates:
        raise MuseStatusParseError(
            "the /status panel renders more than one "
            + ", ".join(sorted(duplicates))
            + " line, so it cannot prove the session it names; refusing rather than "
            "choosing a value"
        )
    missing = [label for label in _BOXED_PANEL_REQUIRED if label not in fields]
    if missing:
        raise MuseStatusParseError(
            "the /status panel is incomplete: missing "
            + ", ".join(sorted(missing))
            + "; a truncated capture is not an observation"
        )

    # The provider/profile continuation row: the first unlabeled row under
    # MODEL, exactly two '·'-separated segments.  It is required identity
    # evidence; a malformed or absent one is refused.  Continuation rows of
    # other labels (trust state, access mode, activity) are chrome.
    profile_segments: Optional[list[str]] = None
    for row in continuations.get("MODEL", []):
        segments = _split_dot_segments(row)
        if segments is not None and len(segments) == 2:
            if profile_segments is not None and segments != profile_segments:
                raise MuseStatusParseError(
                    "the /status panel renders conflicting provider/profile rows "
                    f"{profile_segments!r} and {segments!r}; refusing rather than "
                    "choosing a value"
                )
            profile_segments = segments
    if profile_segments is None:
        raise MuseStatusParseError(
            "the /status panel carries no 'provider · agent profile' continuation "
            "row under MODEL; the profile identity is required evidence and is "
            "never guessed from other rows"
        )

    session_id = fields["SESSION"][0].strip()
    model_segments = _split_dot_segments(fields["MODEL"][0])
    if model_segments is None:
        model, effort = fields["MODEL"][0].strip(), None
    elif len(model_segments) == 2:
        model, effort = model_segments
        if effort not in _MUSE_EFFORT_VOCABULARY:
            raise MuseStatusParseError(
                f"the /status MODEL value carries an unknown effort: {effort!r}; "
                "refusing rather than guessing"
            )
    else:
        raise MuseStatusParseError(
            f"the /status MODEL value is not '<model>' or '<model> · <effort>': "
            f"{fields['MODEL'][0]!r}; refusing rather than guessing"
        )

    usage_match = re.fullmatch(
        r"(\d+)\s+tokens\s*·\s*(\d+)\s+turns(?:\s*·\s*\d+\s+subagents)?", fields["USAGE"][0]
    )
    if usage_match is None:
        raise MuseStatusParseError(
            f"the /status USAGE line is not the expected 'N tokens · N turns' "
            f"shape: {fields['USAGE'][0]!r}"
        )

    parsed = {
        "schema": STATUS_PANEL_SCHEMA,
        "panel_shape": "boxed-0.2",
        "session_id": session_id,
        "model": model,
        "reasoning": effort,
        "agent_profile": profile_segments[1],
        "model_provider": profile_segments[0],
        "directory": fields["WORKSPACE"][0].strip(),
        "run": header_run or "",
        "tokens": int(usage_match.group(1)),
        "turns": int(usage_match.group(2)),
    }
    return parsed


def _footer_observation(rows: Sequence[str]) -> Optional[dict[str, Any]]:
    """The persistent inline footer as a partial route-only observation.

    The 0.2.1+ TUI always renders ``<model> · [<effort> ·] <directory>``
    above the composer, with no session id anywhere on it.  A capture
    whose only recognizable content is that line is real route evidence —
    the pane is up and on the requested model — but it is never readiness:
    the returned observation is explicitly marked partial with
    ``session_id=None``, and every consumer must treat it as "route
    observed, identity absent".

    Two distinct footer-shaped lines are ambiguity and refuse; identical
    repeats converge.
    """
    candidates: list[dict[str, Any]] = []
    for raw in rows:
        row = _strip_panel_row(raw)
        if not row or row.startswith(_BOXED_PANEL_LABELS) or ":" in row.split()[0]:
            continue
        segments = _split_dot_segments(row)
        if segments is None or len(segments) not in (2, 3):
            continue
        if not segments[-1].startswith("/"):
            continue
        model = segments[0]
        directory = segments[-1]
        effort: Optional[str] = None
        if len(segments) == 3:
            effort = segments[1]
            if effort not in _MUSE_EFFORT_VOCABULARY:
                continue
        candidates.append(
            {
                "model": model,
                "reasoning": effort,
                "directory": directory,
            }
        )
    distinct = {tuple(sorted(candidate.items())) for candidate in candidates}
    if not distinct:
        return None
    if len(distinct) > 1:
        raise MuseStatusParseError(
            "the capture renders more than one distinct inline footer "
            f"({len(distinct)} shapes), so it cannot prove the pane's route; "
            "refusing rather than choosing a value"
        )
    values = dict(distinct.pop())
    return {
        "schema": STATUS_PANEL_SCHEMA,
        "partial": True,
        "observation": "route-only-footer",
        "session_id": None,
        "model": values["model"],
        "reasoning": values["reasoning"],
        "directory": values["directory"],
    }


def parse_status_panel(rows: Sequence[str]) -> dict[str, Any]:
    """Parse one ``/status`` capture into typed fields, or refuse.

    Three renderings are understood, tried in order:

    1. The 0.2.1+ boxed panel (detected by its ``┌└`` furniture and
       uppercase labels), parsed strictly by :func:`_parse_boxed_panel`.
    2. The installed 0.1.x labeled panel.  The Model value may carry the
       exact `` (reasoning <effort>)`` suffix, which is split into the
       canonical ``model`` and ``reasoning`` fields; a separate
       ``Reasoning:``/``Reasoning effort:`` row is an additional source
       that must converge with the suffix or be refused as ambiguous.
    3. The persistent inline footer of a 0.2.1+ TUI whose panel has not
       (or does not) render: returned as a PARTIAL observation marked
       ``session_id=None`` — route evidence only, never a ready receipt.

    Raises:
        MuseStatusParseError: The capture is empty, matches no known
            shape, has more than one of any required singleton line
            (including the Session line — the capture cannot prove which
            session the pane runs), carries a malformed reasoning suffix
            or value, lacks a required line, or renders two distinct
            footers.
    """
    if _is_boxed_panel(rows):
        return _parse_boxed_panel(rows)
    try:
        return _parse_labeled_panel(rows)
    except MuseStatusParseError as original:
        partial = _footer_observation(rows)
        if partial is not None:
            return partial
        raise original


def _parse_labeled_panel(rows: Sequence[str]) -> dict[str, Any]:
    """Parse one ``/status`` capture into typed fields, or refuse.

    The Model value may carry the installed `` (reasoning <effort>)``
    suffix, which is split into the canonical ``model`` and ``reasoning``
    fields.  A separate ``Reasoning:``/``Reasoning effort:`` row is an
    additional source that must converge with the suffix or be refused as
    ambiguous.

    Raises:
        MuseStatusParseError: The capture is empty, has no session line,
            has more than one of any required singleton line (including
            the Session line — the capture cannot prove which session the
            pane runs), carries a malformed reasoning suffix or value, or
            lacks a required line.
    """
    fields: dict[str, list[str]] = {}
    reasoning_values: list[str] = []
    for raw in rows:
        row = _strip_panel_row(raw)
        if not row:
            continue
        for label in _REASONING_LABELS:
            if row.startswith(label):
                reasoning_values.append(row[len(label) :].strip())
                break
        else:
            for label in _REQUIRED_LABELS:
                if row.startswith(label):
                    fields.setdefault(label, []).append(row[len(label) :].strip())
                    break

    # Every required field is a singleton; a second one is ambiguity, never
    # a value to pick.  The Session line is the identity and the rest are
    # the route facts, and a capture that cannot prove any one of them
    # proves nothing.
    duplicates = [label for label, values in fields.items() if len(values) > 1]
    if duplicates:
        raise MuseStatusParseError(
            "the /status panel renders more than one "
            + ", ".join(sorted(duplicates))
            + " line, so it cannot prove the session it names; refusing rather than "
            "choosing a value"
        )
    missing = [label for label in _REQUIRED_LABELS if label not in fields]
    if missing:
        raise MuseStatusParseError(
            "the /status panel is incomplete: missing "
            + ", ".join(sorted(missing))
            + "; a truncated capture is not an observation"
        )

    # Model + optional exact reasoning suffix, then the separate-row
    # reasoning values (each validated against the effort vocabulary).
    model, model_reasoning = _split_model_reasoning(fields["Model:"][0])
    label_reasoning: list[Optional[str]] = []
    for value in reasoning_values:
        cleaned = value.strip()
        if not cleaned or cleaned not in _MUSE_EFFORT_VOCABULARY:
            raise MuseStatusParseError(
                f"the /status reasoning value is not a known Muse effort: {value!r}; "
                "refusing rather than guessing"
            )
        label_reasoning.append(cleaned)
    reasoning = _converge_reasoning([model_reasoning, *label_reasoning])

    usage = fields["Token usage:"][0]
    match = re.fullmatch(r"(\d+)\s+tokens\s*/\s*(\d+)\s+turns", usage)
    if match is None:
        raise MuseStatusParseError(
            f"the /status Token usage line is not the expected 'N tokens / N turns' "
            f"shape: {usage!r}"
        )
    tokens, turns = int(match.group(1)), int(match.group(2))

    return {
        "schema": STATUS_PANEL_SCHEMA,
        "panel_shape": "labeled-0.1",
        "session_id": fields["Session:"][0].strip(),
        "model": model,
        "reasoning": reasoning,
        "agent_profile": fields["Agent profile:"][0].strip(),
        "model_provider": fields["Model provider:"][0].strip(),
        "directory": fields["Directory:"][0].strip(),
        "run": fields["Run:"][0].strip(),
        "tokens": tokens,
        "turns": turns,
    }


def validate_discovered_session_id(session_id: Any) -> str:
    """Return a provider-generated session id proven to be a canonical UUID.

    The fresh launch discovers the id from the panel; a session id that is
    not a canonical lowercase UUID cannot be a Muse session identity, so it
    is refused rather than bound.
    """
    if not isinstance(session_id, str) or not session_id:
        raise MuseStatusMismatch(
            f"the /status panel names a session id that is not a canonical UUID: " f"{session_id!r}"
        )
    import uuid as _uuid_module

    try:
        parsed = _uuid_module.UUID(session_id)
    except ValueError as exc:
        raise MuseStatusMismatch(
            f"the /status panel names a session id that is not a canonical UUID: " f"{session_id!r}"
        ) from exc
    if str(parsed) != session_id:
        raise MuseStatusMismatch(
            f"the /status panel names a session id that is not a canonical lowercase "
            f"UUID: {session_id!r}"
        )
    return session_id


def require_pre_task_status(
    parsed: Mapping[str, Any],
    *,
    session_id: Optional[str],
    expected_model: str,
    expected_effort: Optional[str],
    working_directory: str,
    expected_profile_identity: str,
) -> dict[str, Any]:
    """Require the parsed panel to name exactly the claimed pre-task session.

    ``session_id`` is ``None`` on a fresh launch, where the id is
    *discovered*: the panel's session id is validated as a canonical UUID
    and returned as the identity.  When a ``session_id`` is supplied (an
    exact restore), the panel must name exactly it.  Every mismatch raises
    :class:`MuseStatusMismatch` naming the exact field and the observed vs
    required values, so a blocked launch records *which* evidence was
    wrong.  ``expected_effort`` is the requested effort when the route
    selected one (the panel must render it) and ``None`` for a
    provider-default route (no effort line is required and none is claimed
    observed).
    """
    mismatches: list[str] = []

    if parsed.get("partial"):
        raise MuseStatusMismatch(
            "the /status observation is the route-only inline footer "
            f"(model {parsed.get('model')!r}, directory {parsed.get('directory')!r}) "
            "with no session id anywhere on it: it is 'route observed, identity "
            "absent' evidence and can never be a ready receipt"
        )

    observed_session = str(parsed.get("session_id") or "")
    if session_id is None:
        # Fresh launch: the provider generated the id and this panel names
        # it.  Requiring it to be a canonical UUID is the discovery proof.
        validate_discovered_session_id(observed_session)
        session_matches = True
    else:
        session_matches = observed_session == session_id
        if not session_matches:
            mismatches.append(
                f"session: the panel names {observed_session!r}, not the expected "
                f"{session_id!r}"
            )

    observed_model = str(parsed.get("model") or "")
    model_matches = bool(expected_model) and observed_model == expected_model
    if not model_matches:
        mismatches.append(
            f"model: the panel names {observed_model!r}, not the requested " f"{expected_model!r}"
        )

    observed_effort = parsed.get("reasoning")
    effort_matches: bool
    if expected_effort and expected_effort != provider_contracts.EFFORT_PROVIDER_DEFAULT:
        effort_matches = bool(observed_effort) and str(observed_effort) == expected_effort
        if not effort_matches:
            mismatches.append(
                f"effort: the panel renders reasoning {observed_effort!r}, not the "
                f"requested {expected_effort!r}"
            )
    else:
        # A provider-default route requests no effort (the sentinel is not
        # an effort to observe); the panel may render the provider's own
        # default, which is not an observed request.
        effort_matches = True

    observed_profile = str(parsed.get("agent_profile") or "")
    profile_matches = observed_profile == expected_profile_identity
    if not profile_matches:
        mismatches.append(
            f"agent profile: the panel names {observed_profile!r}, not the expected "
            f"{expected_profile_identity!r}"
        )

    observed_provider = str(parsed.get("model_provider") or "")
    provider_matches = observed_provider == "meta"
    if not provider_matches:
        mismatches.append(f"provider: the panel names {observed_provider!r}, not 'meta'")

    observed_directory = str(parsed.get("directory") or "")
    # The panel renders the workspace the way muse names it, which
    # abbreviates the home directory to ``~`` (live-proven: the pane shows
    # ``~/Projects/cao-smoke`` while the bound claim is the absolute path).
    # Compare expanded paths; the rendered abbreviation is not a mismatch.
    directory_matches = (
        observed_directory == working_directory
        or os.path.expanduser(observed_directory) == working_directory
        or observed_directory == os.path.expanduser(working_directory)
        or os.path.expanduser(observed_directory) == os.path.expanduser(working_directory)
    )
    if not directory_matches:
        mismatches.append(
            f"cwd: the panel names {observed_directory!r}, not the bound " f"{working_directory!r}"
        )

    run = str(parsed.get("run") or "")
    idle = run == PRE_TASK_RUN_STATE
    if not idle:
        mismatches.append(f"run state: the panel reads {run!r}, not {PRE_TASK_RUN_STATE!r}")

    tokens = parsed.get("tokens")
    turns = parsed.get("turns")
    zero_turns = tokens == 0 and turns == 0
    if not zero_turns:
        mismatches.append(
            f"pre-task usage: the panel reads {tokens} tokens / {turns} turns, not "
            f"{PRE_TASK_TOKEN_USAGE[0]} tokens / {PRE_TASK_TOKEN_USAGE[1]} turns"
        )

    if mismatches:
        raise MuseStatusMismatch(
            "the /status panel does not describe the claimed pre-task session: "
            + "; ".join(mismatches)
        )

    return {
        "schema": STATUS_PANEL_SCHEMA,
        "session_matches": True,
        "model_matches": True,
        "effort_matches": True,
        "profile_matches": True,
        "provider_matches": True,
        "directory_matches": True,
        "idle": True,
        "zero_turns": True,
        "observed": {
            "session_id": observed_session,
            "model": observed_model,
            "effort": observed_effort,
            "agent_profile": observed_profile,
            "model_provider": observed_provider,
            "directory": observed_directory,
            "run": run,
            "tokens": tokens,
            "turns": turns,
        },
    }
