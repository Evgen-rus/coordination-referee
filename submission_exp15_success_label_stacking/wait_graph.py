"""Exp13 - explicit wait-dependency graph for `deadlock`.

THE ONE HYPOTHESIS
------------------
    An explicit representation of WHO waits for WHOM adds relational
    information that the current 300-feature Exp08 representation does not
    contain, and improves the official composite.

The knowledge audit of `deadlock` established, from the official ontology and
from the train corpus only:

  * ``docs/ontology.md`` requires "A waits for B's subtask, B waits for A's",
    a status exchange, and no progress afterwards;
  * the generator emits exactly SIX waiting message templates, and **all six
    name the awaited agent explicitly**;
  * a reciprocal same-subtask wait pair occurs in 82.9% of true `deadlock`
    runs versus 45.6-47.7% of every other class;
  * that relational structure is only recoverable from the current features
    through aggregate proxies (``share_status``, r ~ 0.72).

WHAT THIS MODULE IS NOT
-----------------------
It is not a lexical status detector.  ``share_status``, ``kw_wait``,
``tail_status_share``, ``g_pingpong`` and ``g_reciprocity`` already exist in
the 300 features; re-deriving them would be a duplicate, not a new
representation.  Everything computed here is a property of the DIRECTED
WAIT-DEPENDENCY RELATION ``sender -> awaited_agent``, which the aggregate
communication-graph features do not encode because they count message edges
of every type together and never parse *who is being waited for*.

FIXED PARSER (frozen before any model was fitted)
-------------------------------------------------
Six template families, one compiled regex each.  Each returns
``(sender, awaited_agent, subtask_or_None, turn)``.  The regex list is a
literal transcription of the six families named in the audit and is asserted
to be exactly six entries by ``test_features.py``.  No regex is added,
removed or loosened after any CV number was seen.

LEAKAGE
-------
The only inputs read are ``messages`` entries and their fields ``t``, ``from``
and ``text``.  ``label``, ``success``, ``fault_turn`` and any prediction are
never touched; ``test_features.py`` proves it by target perturbation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# The six official template families.  Order is fixed and matters only for
# the unit test that asserts the count.
# --------------------------------------------------------------------------
TEMPLATE_NAMES = (
    "blocked_on",              # Blocked on <agent>: ...
    "waiting_for_deliver",     # Waiting for <agent> to deliver ...
    "cannot_proceed_until",    # Cannot proceed ... until <agent> finishes ...
    "on_hold_from",            # On hold: <subtask> from <agent> has not arrived
    "still_holding_for",       # Still holding for <agent> on ...
    "step_depends_on",         # My step depends on <agent> closing ...
)

# Agent ids in the corpus are ``a<digits>``.  Bounding the token to that shape
# keeps the parser from firing on ordinary prose that happens to contain a
# lowercase letter followed by digits (e.g. "s3", "h2" inside a goal word).
_AG = r"(a[0-9]+)"
_SUB = r"([a-z][a-z0-9_]*)"

TEMPLATE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bblocked on %s\b" % _AG, re.I),
    re.compile(r"\bwaiting for %s to deliver\b" % _AG, re.I),
    re.compile(r"\bcannot proceed\b.*?\buntil %s finishes\b" % _AG, re.I),
    re.compile(r"\bon hold:.*?\bfrom %s\b" % _AG, re.I),
    re.compile(r"\bstill holding for %s on\b" % _AG, re.I),
    re.compile(r"\bmy step depends on %s closing\b" % _AG, re.I),
)

# Subtask extraction, one per family, tried only on the family that matched.
# Each pattern returns None when the text does not name a subtask; the wait
# event then still contributes an agent-level edge but never invents a
# subtask, as required by the experiment specification.
SUBTASK_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bblocked on %s:.*?\bi need %s\b" % (_AG, _SUB), re.I),
    re.compile(r"\bwaiting for %s to deliver %s\b" % (_AG, _SUB), re.I),
    re.compile(r"\buntil %s finishes %s\b" % (_AG, _SUB), re.I),
    re.compile(r"\bon hold: %s from %s\b" % (_SUB, _AG), re.I),
    re.compile(r"\bholding for %s on %s\b" % (_AG, _SUB), re.I),
    re.compile(r"\bdepends on %s closing %s\b" % (_AG, _SUB), re.I),
)

# Capture-group index of the SUBTASK, per family.  It is NOT always group 2:
# ``on_hold_from`` writes the subtask before the agent ("On hold: X from a4"),
# so there the subtask is group 1.  Making this explicit removes the whole class
# of "wrong group index" bugs and documents the textual order per template.
SUBTASK_GROUP: Tuple[int, ...] = (2, 2, 2, 1, 2, 2)

# A message only counts as a wait if the generator marked it `status`.  This is
# an observable schema field, not a keyword guess, and keeps the parser from
# reacting to an `inform` message that happens to mention waiting.
STATUS_TYPE = "status"


def parse_wait(msg: Dict[str, Any]) -> Optional[Tuple[str, str, Optional[str], int]]:
    """Return ``(sender, awaited_agent, subtask_or_None, turn)`` or ``None``.

    ``None`` means "this message is not an observable wait".  A matched message
    whose subtask cannot be read yields ``subtask=None`` rather than a guess.
    """
    if msg.get("type") != STATUS_TYPE:
        return None
    text = str(msg.get("text") or "")
    if not text:
        return None
    for i, pat in enumerate(TEMPLATE_PATTERNS):
        m = pat.search(text)
        if not m:
            continue
        sender = str(msg.get("from") or "")
        awaited = m.group(1).lower()
        sm = SUBTASK_PATTERNS[i].search(text)
        subtask = sm.group(SUBTASK_GROUP[i]).lower() if sm else None
        if subtask in ("a", "and", "the", "it"):
            subtask = None
        return sender, awaited, subtask, int(msg.get("t", -1))
    return None


def wait_events(run: Dict[str, Any]) -> List[Tuple[str, str, Optional[str], int]]:
    """All observable wait events of one run, in message order."""
    out = []
    for m in run.get("messages", []) or []:
        ev = parse_wait(m)
        if ev is not None:
            out.append(ev)
    return out


FEATURE_NAMES = (
    "wg_n_wait_edges",
    "wg_n_unique_wait_edges",
    "wg_n_recip_pairs",
    "wg_has_recip_pair",
    "wg_recip_wait_share",
    "wg_n_same_subtask_recip_pairs",
    "wg_has_same_subtask_recip_pair",
    "wg_earliest_recip_pos_rel",
    "wg_max_pair_wait_events",
    "wg_max_pair_repeat_after_onset",
    "wg_max_pair_span_rel",
    "wg_n_agents_in_recip_pairs",
)
N_FEATURES = len(FEATURE_NAMES)


def extract_wait_graph_features(run: Dict[str, Any]) -> Dict[str, float]:
    """The fixed 12-feature wait-graph block.

    Returns exactly ``FEATURE_NAMES``; a run with no wait message yields all
    zeros, which is a legitimate state (no wait relation observed) rather than
    a missing value.
    """
    msgs = run.get("messages", []) or []
    n_msgs = len(msgs)
    evs = wait_events(run)
    f = {k: 0.0 for k in FEATURE_NAMES}
    f["wg_n_wait_edges"] = float(len(evs))
    if not evs:
        return f

    edges = [(a, b) for a, b, _s, _t in evs]
    f["wg_n_unique_wait_edges"] = float(len(set(edges)))
    f["wg_recip_wait_share"] = float(len(edges)) / float(n_msgs)

    edge_set = set(edges)
    pairs = set()
    for a, b in edge_set:
        if a != b and (b, a) in edge_set:
            pairs.add((a, b) if a < b else (b, a))
    f["wg_n_recip_pairs"] = float(len(pairs))
    f["wg_has_recip_pair"] = 1.0 if pairs else 0.0

    # same-subtask reciprocal pairs: both directions explicitly name one and
    # the same subtask.  An event without a readable subtask cannot contribute
    # to this test and is simply not counted on its side of the pair.
    fwd = {}
    bwd = {}
    for a, b, s, t in evs:
        if s is None:
            continue
        fwd.setdefault((a, b), set()).add(s)
        bwd.setdefault((b, a), set()).add(s)
    same = {p for p in pairs if fwd.get(p, set()) & bwd.get(p, set())}
    f["wg_n_same_subtask_recip_pairs"] = float(len(same))
    f["wg_has_same_subtask_recip_pair"] = 1.0 if same else 0.0

    if not pairs:
        return f

    # A pair becomes "complete" at the LATER of its two FIRST wait turns - that
    # is the ontology's "first of the paired waiting messages" onset.  Using the
    # LAST turn of the pair here would make every repeat count as "before
    # onset", which silently zeroes wg_max_pair_repeat_after_onset.
    denom = float(max(1, n_msgs - 1))
    turn_of = {}
    first_of = {}
    for p in pairs:
        turn_of[p] = sorted(t for a, b, _s, t in evs
                            if (a, b) == p or (a, b) == (p[1], p[0]))
        fwd = [t for a, b, _s, t in evs if (a, b) == p]
        bwd = [t for a, b, _s, t in evs if (a, b) == (p[1], p[0])]
        first_of[p] = max(min(fwd), min(bwd))
    onset = {}
    span = {}
    count = {}
    agents = set()
    for a, b, _s, t in evs:
        p = (a, b) if a < b else (b, a)
        if p not in pairs:
            continue
        agents.update(p)
        count[p] = count.get(p, 0) + 1
        lo, hi = onset.get(p, (t, t))
        onset[p] = (min(lo, t), max(hi, t))
    for p, (lo, hi) in onset.items():
        span[p] = hi - lo

    complete_at = first_of
    f["wg_earliest_recip_pos_rel"] = float(min(complete_at.values())) / denom

    active = max(count.items(), key=lambda kv: (kv[1], kv[0]))
    f["wg_max_pair_wait_events"] = float(active[1])
    f["wg_max_pair_span_rel"] = float(span[active[0]]) / denom

    on = complete_at[active[0]]
    f["wg_max_pair_repeat_after_onset"] = float(sum(
        1 for t in turn_of[active[0]] if t > on))

    f["wg_n_agents_in_recip_pairs"] = float(len(agents))
    return f


def build_matrix(runs) -> np.ndarray:
    """``(n_runs, 12)`` float64 matrix in ``FEATURE_NAMES`` order."""
    out = np.zeros((len(runs), N_FEATURES), dtype=np.float64)
    for i, run in enumerate(runs):
        f = extract_wait_graph_features(run)
        out[i] = [f[k] for k in FEATURE_NAMES]
    if not np.isfinite(out).all():
        raise ValueError("wait-graph block contains non-finite values")
    return out
