"""Experiment protocol guard - read-only.  No DB, no service, no orchestration.

WHAT THIS IS
------------
A guard, not a framework.  It answers one question before a new experiment is
invented:

    "given the history in ``experiments/results.csv``, is starting Exp<N+1> a
     legal move right now, and if not, what is missing?"

It derives everything from the two artifacts that already exist and that
Exp09/Exp10 already write:

  * ``experiments/results.csv``               - the single experiment history
  * ``experiments/<exp>/results.json``         - each run's own verdict, which
    already contains ``promotion_gate`` and ``reproduction_gate``

Nothing is recomputed here.  In particular this module does NOT re-implement
the reproduction gate, the cross-fit, or the promotion gate: those belong to the
experiment that declared them, and duplicating them here would create a second
source of truth that can silently disagree with the first.  This guard only
*reads* the verdict each experiment already recorded.

WHY IT EXISTS
-------------
Exp09 and Exp10 both ended in STOP.  The protocol that produced those two
verdicts lives entirely inside each run's own module docstring, so an agent
starting Exp11 has nothing that forces it to follow the same discipline.  The
missing piece is the rule that says what happens when a hillclimb stalls - that
is the part implemented below.

STATE OF THE REPO WHEN THIS WAS WRITTEN
---------------------------------------
    exp08  PROMOTE (current best)  ->  sub08 shipped
    exp09  STOP   (composite -0.00021, 1/3 folds)
    exp10  STOP   (composite -0.00090, 0/3 folds)

Two consecutive STOPS, so the stall rule is NOT yet blocking: Exp11 may be run
as the third experiment of this cycle.  If Exp11 also STOPS, Exp12 may not be
invented until a root-cause review of exp09/exp10/exp11 exists.
"""

from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_CSV = os.path.join(HERE, "results.csv")

# Verdicts used in ``results.csv`` column 9.  These tokens are ALREADY the
# project's own vocabulary - the guard does not invent a status field, it reads
# the one that has been written since exp01.
STOP_VERDICTS = frozenset({"not_promising", "not_promoting"})
PROMOTE_VERDICTS = frozenset({"promising", "promoting"})

# The stall rule, as one number so the protocol document and the code cannot
# drift apart (a test pins both to 3).
STALL_THRESHOLD = 3

# ``results.csv`` is positional and has no header row.  Column indices are fixed
# by exp01 and were never changed; exp08 stores the composite delta in the
# last-metric column, so no column here is reinterpreted as a composite.
COL_ID, COL_NAME, COL_VERDICT, COL_RESULTS = 0, 1, 8, 10


# ---------------------------------------------------------------------------
# 1. history
# ---------------------------------------------------------------------------

class Row(object):
    """One ``results.csv`` line, read-only, with the columns this guard uses."""

    __slots__ = ("exp", "name", "verdict", "results_rel", "line_no")

    def __init__(self, exp, name, verdict, results_rel, line_no):
        self.exp = exp
        self.name = name
        self.verdict = verdict
        self.results_rel = results_rel
        self.line_no = line_no

    def is_experiment(self):
        """True for a research run; False for a packaged production build.

        ``sub*`` rows are builds of an already-promoted system, not new
        hypotheses, so they neither count toward a hillclimb nor reset it.
        """
        return self.exp.startswith("exp")

    def __repr__(self):
        return "<%s %s %s>" % (self.exp, self.name, self.verdict)


def read_history(results_csv=None):
    """Parse ``results.csv`` into :class:`Row` objects, in file order."""
    path = results_csv or RESULTS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError("experiment history not found: %s" % path)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for i, raw in enumerate(csv.reader(fh), start=1):
            if not raw or not raw[0].strip():
                continue
            if len(raw) <= COL_RESULTS:
                raise ValueError(
                    "results.csv line %d has %d fields, expected at least %d - "
                    "the history schema changed and the guard must be updated"
                    % (i, len(raw), COL_RESULTS + 1))
            rows.append(Row(raw[COL_ID].strip(), raw[COL_NAME].strip(),
                            raw[COL_VERDICT].strip(), raw[COL_RESULTS].strip(), i))
    return rows


def exp_dir(row, root=None):
    """Directory of one experiment, derived from its own ``results.csv`` path."""
    rel = row.results_rel.replace("\\", "/")
    return os.path.join(root or ROOT, os.path.dirname(rel))


# ---------------------------------------------------------------------------
# 2. verdict - two independent signals, and they must agree
# ---------------------------------------------------------------------------

def own_gate(row, root=None):
    """The run's OWN recorded promotion gate: True / False / None (absent).

    ``None`` means the experiment predates the promotion-gate convention
    (exp01-exp08) or has no ``results.json``; the CSV verdict then stands alone.
    """
    path = os.path.join(exp_dir(row, root), "results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    gate = data.get("promotion_gate")
    if isinstance(gate, dict) and "passed" in gate:
        return bool(gate["passed"])
    return None


def classify(row, root=None):
    """Return ``(kind, reason)`` with kind in promote / stop / conflict.

    Two independent sources are compared - the ``results.csv`` verdict and the
    run's own ``promotion_gate`` - because a row that claims STOP while its own
    results.json says the gate PASSED is drift in the history, and drift that
    silently counts as a promotion would unblock a stalled hillclimb.
    """
    csv_stop = row.verdict in STOP_VERDICTS
    csv_promote = row.verdict in PROMOTE_VERDICTS
    if not csv_stop and not csv_promote:
        return "stop", ("unrecognised verdict %r - treated as STOP so a broken "
                        "row can never unblock a stall" % row.verdict)
    gate = own_gate(row, root)
    csv_kind = "stop" if csv_stop else "promote"
    if gate is None:
        return csv_kind, "results.csv verdict %r (no promotion_gate recorded)" \
            % row.verdict
    if (gate is False) == csv_stop:
        return csv_kind, "results.csv %r agrees with promotion_gate=%s" \
            % (row.verdict, gate)
    return "conflict", ("results.csv says %r but promotion_gate.passed=%s - "
                        "fix the history before trusting it"
                        % (row.verdict, gate))


# ---------------------------------------------------------------------------
# 3. current best and the stall counter
# ---------------------------------------------------------------------------

def experiments(rows):
    """Only the research runs, in file order."""
    return [r for r in rows if r.is_experiment()]


def current_best(rows, root=None):
    """The most recent promoted experiment - the baseline Exp<N+1> must copy.

    A failed experiment can never become the baseline: the loop below only ever
    returns a row classified ``promote``, so a STOP is structurally excluded
    rather than merely discouraged.
    """
    promoted = [r for r in experiments(rows) if classify(r, root)[0] == "promote"]
    if not promoted:
        return None
    return promoted[-1]


def consecutive_stops(rows, root=None):
    """The unbroken tail of STOPs, newest last."""
    tail = []
    for row in reversed(experiments(rows)):
        kind, _ = classify(row, root)
        if kind == "promote":
            break
        tail.append(row)                      # conflict counts as STOP, fail-safe
    return list(reversed(tail))


def stall_state(rows=None, root=None):
    """The stalled-hillclimb rule, as data."""
    rows = rows if rows is not None else read_history()
    tail = consecutive_stops(rows, root)
    n = len(tail)
    return {
        "n_consecutive_stops": n,
        "threshold": STALL_THRESHOLD,
        "stopped": [r.exp for r in tail],
        "blocking": n >= STALL_THRESHOLD,
        "remaining_before_block": max(0, STALL_THRESHOLD - n),
        "review_ok": root_cause_review_ok(tail, root),
    }


# ---------------------------------------------------------------------------
# 4. the root-cause review - a file on disk, not a claim made in chat
# ---------------------------------------------------------------------------

REVIEW_FILE = "ROOT_CAUSE_REVIEW.md"
#: Every key the review header must carry.  ``status`` is checked separately
#: because its required value is the word "complete".
REVIEW_KEYS = ("covers", "hypotheses", "transfer", "headroom", "noise",
               "warrant")
#: The five questions the review must answer, mapped to their header key.
REVIEW_QUESTIONS = {
    "hypotheses": "which hypotheses were tested, in one line each",
    "transfer": "why they did not transfer to held-out rows",
    "headroom": "where measurable headroom is still left",
    "noise": "which directions already look noise-dominated",
    "warrant": "whether there is a warrant for the next experiment at all",
}
#: The same five, ordered, as the guard prints them for the agent.
REVIEW_QUESTION_KEYS = ("hypotheses", "transfer", "headroom", "noise", "warrant")


def review_path(root=None):
    return os.path.join(root or ROOT, "experiments", REVIEW_FILE)


def parse_review(text):
    """Parse the leading ``key: value`` block of the review.

    Deliberately not a full markdown parser: the machine-checkable part is a
    short ASCII block at the top, and the prose below it is for humans.
    """
    meta = {}
    for line in text.splitlines():
        if line.strip() == "":
            if meta:
                break                       # header block ended
            continue
        if line.lstrip().startswith("#"):
            if meta:
                break
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def review_defects(tail, root=None):
    """Every reason the root-cause review is missing or unusable. Empty == ok."""
    defects = []
    path = review_path(root)
    if not os.path.exists(path):
        return ["no %s - write it before the next experiment is invented"
                % REVIEW_FILE]
    with open(path, encoding="utf-8") as fh:
        meta = parse_review(fh.read())
    for key in REVIEW_KEYS:
        if not meta.get(key):
            defects.append("review is missing the %r key (%s)"
                           % (key, REVIEW_QUESTIONS.get(key, "required")))
    if meta.get("status", "").lower() != "complete":
        defects.append("review status is %r, not 'complete' - a draft does not "
                       "unblock anything" % meta.get("status", ""))
    covered = [t.strip() for t in meta.get("covers", "").replace("[", "")
               .replace("]", "").split(",") if t.strip()]
    expected = [r.exp for r in tail]
    if covered != expected:
        defects.append("review covers %r but the stalled tail is %r - a stale "
                       "review must not satisfy a new stall"
                       % (covered, expected))
    return defects


def root_cause_review_ok(tail=None, root=None):
    if tail is None:
        tail = consecutive_stops(read_history(), root)
    return not review_defects(tail, root)


# ---------------------------------------------------------------------------
# 5. the check itself
# ---------------------------------------------------------------------------

def check(rows=None, root=None):
    """Full protocol verdict. Returns a dict; ``main()`` prints it.

    ``violations`` are things that are wrong NOW.  ``blocking`` is the only
    reason a new experiment may not be started.
    """
    rows = rows if rows is not None else read_history()
    root = root or ROOT
    exps = experiments(rows)
    violations = []
    conflicts = []

    for row in exps:
        kind, reason = classify(row, root)
        if kind == "conflict":
            conflicts.append("%s: %s" % (row.exp, reason))
    violations.extend(conflicts)

    best = current_best(rows, root)
    if best is None:
        violations.append("no promoted experiment in the history - there is no "
                          "baseline to reproduce")

    # a failed experiment must not have left a production build behind
    for row in exps:
        gate = own_gate(row, root)
        if gate is False:
            violations.extend(stray_builds(row, root))

    state = stall_state(rows, root)
    blocking, reasons = [], []
    if state["blocking"] and not state["review_ok"]:
        blocking.append(
            "STALLED HILLCLIMB: %d consecutive STOPS (%s) >= threshold %d."
            % (state["n_consecutive_stops"], ", ".join(state["stopped"]),
               STALL_THRESHOLD))
        reasons.extend(review_defects(
            consecutive_stops(rows, root), root))

    return {
        "current_best": best.exp if best else None,
        "current_best_dir": (os.path.relpath(exp_dir(best, root), root)
                             if best else None),
        "experiments": [r.exp for r in exps],
        "stall": state,
        "violations": violations,
        "blocking": blocking,
        "block_reasons": reasons,
        "can_start_next_experiment": not blocking and not conflicts,
    }


def build_marker(row):
    """The numeric id a production build would carry: ``exp09`` -> ``09``.

    ``exp06b`` deliberately yields ``06``: the ``b`` suffix is a validation-only
    sub-experiment, so a build of ``exp06`` legitimately matches it.
    """
    digits = "".join(ch for ch in row.exp if ch.isdigit())
    return digits if len(digits) == 2 else None


def stray_builds(row, root=None):
    """A STOP must not have left a submission directory or a ZIP behind.

    This is the rule "a failed experiment produces no production build", checked
    against the repository root rather than trusted.  It is deliberately narrow:
    it only fires for a build whose name carries the STOPping experiment's own
    number, so a legitimate build of the current best is never reported.
    """
    root = root or ROOT
    marker = build_marker(row)
    if marker is None:
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("submission"):
            continue
        if marker in name:
            out.append("%s STOPS but %s exists - a failed experiment must not "
                       "ship a production build" % (row.exp, name))
    return out


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def _fmt(report):
    stall = report["stall"]
    out = []
    out.append("=" * 72)
    out.append("EXPERIMENT PROTOCOL GUARD")
    out.append("=" * 72)
    out.append("current best / baseline to reproduce : %s"
               % (report["current_best"] or "NONE"))
    if report["current_best_dir"]:
        out.append("  %s" % report["current_best_dir"])
    out.append("experiments so far                  : %s"
               % ", ".join(report["experiments"]))
    out.append("consecutive STOPS                   : %d / %d  (%s)"
               % (stall["n_consecutive_stops"], stall["threshold"],
                  ", ".join(stall["stopped"]) or "none"))
    if stall["review_ok"]:
        out.append("root-cause review                  : present and current")
    else:
        out.append("root-cause review                  : absent or stale")
        out.append("  required: experiments/%s, with this header:" % REVIEW_FILE)
        out.append("      status: complete")
        out.append("      covers: [%s]" % ", ".join(stall["stopped"]))
    for key in REVIEW_QUESTION_KEYS:
        out.append("      %-12s %s" % (key, REVIEW_QUESTIONS[key]))
        out.append("      -> one line answering it")
    out.append("")
    for v in report["violations"]:
        out.append("  VIOLATION  %s" % v)
    for b in report["blocking"]:
        out.append("  BLOCKED    %s" % b)
    for r in report["block_reasons"]:
        out.append("            - %s" % r)
    out.append("")
    if report["can_start_next_experiment"]:
        out.append("VERDICT: a new experiment MAY be started.")
        if stall["remaining_before_block"]:
            out.append("         %d more STOP(s) and the root-cause review becomes "
                       "mandatory." % stall["remaining_before_block"])
    else:
        out.append("VERDICT: DO NOT invent the next experiment.")
        out.append("         Complete the root-cause review first, then re-run this "
                   "guard.")
    out.append("")
    out.append("The guard does not choose the hypothesis and does not create the")
    out.append("experiment. It only refuses to let a stalled hillclimb continue")
    out.append("unreviewed. Protocol: experiments/PROTOCOL.md")
    return "\n".join(out)


def main():
    report = check()
    text = _fmt(report)
    print(text)
    return 0 if report["can_start_next_experiment"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
