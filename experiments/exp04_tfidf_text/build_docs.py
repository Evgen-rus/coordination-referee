"""Experiment 04: build one text document per run for TF-IDF.

A document is assembled from target-free parts of the run only:

  [GOAL] <goal text>
  [HANDOFF] [STATUS] [STATE] [FINAL] [SYSTEM] [INFORM]   protocol type markers
  [SUB] <intent>        intent name as a lexical token, per message
  [NOINTENT]            emitted for every message whose intent is missing
  [DELIVERED]           a handoff that carries artifact refs
  [TOOL] <tool>         declared capability name
  [ROLE] <role>         agent role, not agent id
  [ART]/[ARTSTATUS]/[STATEKEY]/[STATEOP]
  [NOKEYWORD] [UNANSWERED_MANY] [REASSIGN] [TIE] [RESTART] [DUP] [DEAD]
  [CONFLICT] [DRIFT] [LOOP]      structural events as presence tokens

Sender ids (``a1``, ``a2``, ...) are deliberately NOT emitted: they are arbitrary
identifiers and a linear model would memorise them rather than learn language.
Agent *roles* are emitted instead, because roles are meaningful and stable.

All structural tokens are computed from protocol structure alone and are purely
target-free.  Never reads ``label``, ``success`` or ``fault_turn``.

Run:  .\\.venv\\Scripts\\python.exe experiments\\exp04_tfidf_text\\build_docs.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from features import _norm, _tokens   # noqa: E402  (official baseline, read-only)

# rare shift domain named in docs/data_schema.md; a diagnostic slice only
COMPLIANCE_RE = re.compile(
    r"compliance|audit|sweep|evidence\s*pack|red\s*flag|policy\s*update|\bdpa\b|\bmsa\b",
    re.I)


def domain_proxy(goal: str) -> str:
    """Leakage-free domain bucket, used only for the shift diagnostic."""
    return "compliance_audit" if COMPLIANCE_RE.search(goal or "") else "main_domain"


def _state_events(state: List[Dict]) -> Tuple[int, int, bool]:
    """Return (n_value_flips, n_alternating_flips, any_override)."""
    per_key: Dict[str, List[Dict]] = defaultdict(list)
    override = False
    for s in sorted(state, key=lambda x: int(x.get("t", 0))):
        per_key[s.get("key")].append(s)
        if str(s.get("op") or "").lower() == "override":
            override = True
    flips = alt = 0
    for _k, evs in per_key.items():
        vals = [e.get("value") for e in evs]
        for j in range(1, len(vals)):
            if vals[j] != vals[j - 1]:
                flips += 1
        for j in range(2, len(vals)):
            # a value that returns to its earlier state: oscillation
            if vals[j] != vals[j - 1] and vals[j - 1] != vals[j - 2] and vals[j] == vals[j - 2]:
                alt += 1
    return flips, alt, override


def build_doc(run: Dict[str, Any]) -> str:
    goal = run.get("goal") or ""
    msgs = run.get("messages") or []
    agents = run.get("agents") or []
    arts = run.get("artifacts") or []
    state = run.get("shared_state") or []

    roles = {a.get("id"): str(a.get("role") or "unknown").lower()
             for a in agents if a.get("id") is not None}

    # ---- intent bookkeeping (target-free) ----
    per_intent: Dict[str, List[int]] = defaultdict(list)
    owners: Dict[str, set] = defaultdict(set)
    for i, m in enumerate(msgs):
        it = (m.get("intent") or "").strip()
        if it:
            per_intent[it].append(i)
            owners[it].add(m.get("from", ""))
    never_delivered, restart = set(), set()
    for it, idxs in per_intent.items():
        if not any((msgs[i].get("refs") or []) for i in idxs) and \
                any(msgs[i].get("type") == "handoff" for i in idxs):
            never_delivered.add(it)
        if len({msgs[i].get("from") for i in idxs if msgs[i].get("from")}) > 1:
            restart.add(it)
    multi_owner = {it for it, o in owners.items() if len(o) > 1}

    # ---- artifact duplication ----
    h2sub: Dict[str, List[str]] = defaultdict(list)
    for a in arts:
        if a.get("hash"):
            h2sub[str(a["hash"])].append(str(a.get("subtask") or ""))
    dup_hash = any(len([s for s in subs if s]) > 1 or
                   (len(subs) > 1 and len(set(subs)) == 1 and len(subs) > 1)
                   for subs in h2sub.values())

    # ---- shared state ----
    flips, alt_flips, override = _state_events(state)

    # ---- goal-overlap collapse ----
    gt = _tokens(goal)
    goal_zero = False
    if gt:
        for i in range(len(msgs) - 1):
            if not (gt & _tokens(msgs[i].get("text", "") or "")) \
               and not (gt & _tokens(msgs[i + 1].get("text", "") or "")):
                goal_zero = True
                break

    # ---- repeat streak ----
    norms = [_norm(m.get("text", "") or "") for m in msgs]
    streak = sum(1 for i in range(1, len(norms)) if norms[i] and norms[i] == norms[i - 1])

    # ---- assemble ----
    parts: List[str] = ["[GOAL] " + _norm(goal)]
    for m in msgs:
        tp = (m.get("type") or "other").upper()
        parts.append("[" + tp + "]")
        it = (m.get("intent") or "").strip()
        parts.append(("[SUB] " + _norm(it).replace(" ", "_")) if it else "[NOINTENT]")
        parts.append("[ROLE] " + (roles.get(m.get("from", "")) or "unknown"))
        tx = _norm(m.get("text", "") or "")
        parts.append(tx if tx else "[NOTEXT]")
        tl = (m.get("tool") or "").strip()
        if tl:
            parts.append("[TOOL] " + _norm(tl).replace(" ", "_"))
        if (m.get("refs") or []) and (m.get("type") == "handoff"):
            parts.append("[DELIVERED]")
    for a in arts:
        t = _norm(a.get("type") or "").replace(" ", "_")
        if t:
            parts.append("[ART] " + t)
        st = str(a.get("status") or "").lower()
        if st:
            parts.append("[ARTSTATUS] " + st)
    for s in state:
        k = _norm(s.get("key") or "").replace(" ", "_")
        if k:
            parts.append("[STATEKEY] " + k)
        op = str(s.get("op") or "").lower()
        if op:
            parts.append("[STATEOP] " + op)

    # structural events as presence tokens (deduplicated, not spammed)
    ev = set()
    if never_delivered:
        ev.add("NOKEYWORD")
    if len(never_delivered) >= 2:
        ev.add("UNANSWERED_MANY")
    if restart:
        ev.add("REASSIGN")
    if multi_owner:
        ev.add("TIE")
    if dup_hash:
        ev.add("DUP")
    if flips:
        ev.add("DEAD")
    if alt_flips or override:
        ev.add("CONFLICT")
    if goal_zero:
        ev.add("DRIFT")
    if streak >= 2:
        ev.add("LOOP")
    parts.extend("[" + t + "]" for t in sorted(ev))
    return " ".join(p for p in parts if p)


def build_all(df: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray]:
    from features import parse_run
    docs, doms, nchars = [], [], []
    for r in df.to_dict("records"):
        docs.append(build_doc(parse_run(r)))
        doms.append(domain_proxy(r.get("goal", "")))
        nchars.append(len(docs[-1]))
    return docs, doms, np.array(nchars)


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    docs, doms, nc = build_all(train)
    print("runs=%d  doc chars: mean=%.0f median=%.0f p95=%.0f max=%d"
          % (len(docs), nc.mean(), np.median(nc), np.percentile(nc, 95), nc.max()))
    print("domain proxy: %s" % dict(Counter(doms)))
    with open(os.path.join(HERE, "docs_train.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f)
    print("wrote docs_train.json (%.1f MB)"
          % (os.path.getsize(os.path.join(HERE, "docs_train.json")) / 1e6))
    print("")
    print("=== example document (run 0) ===")
    print(docs[0][:900])
    print("")
    j = int(np.argmax([d.count("[NOINTENT]") for d in docs]))
    print("=== no-intent run (index %d, NOINTENT=%d) ===" % (j, docs[j].count("[NOINTENT]")))
    print(docs[j][:900])
