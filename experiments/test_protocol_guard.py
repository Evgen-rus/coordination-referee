"""Tests for the experiment protocol guard.

Deliberately few.  A test that cannot fail on a real regression is worse than
no test, so each case here pins a rule that the protocol depends on and that a
future edit could plausibly break:

  * the stall threshold is 3, and PROTOCOL.md says 3 (doc/code drift guard);
  * a failed experiment can never become the baseline;
  * ``results.csv`` and the run's own ``promotion_gate`` must agree, and a
    disagreement is fail-safe - it counts as STOP, never as a promotion;
  * a root-cause review unblocks a stall only while it is current;
  * the guard is READ-ONLY: running it must not touch the history.

Every case is a PROPERTY of the protocol, not a snapshot of today's history.
Nothing here asserts a particular experiment number, a particular current
best, or a particular STOP-tail: those change the moment Exp11 lands, and a
test that breaks then would train everyone to ignore the suite.  The recorded
state lives in PROTOCOL.md as a dated snapshot and in ``results.csv``, and the
guard derives the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import protocol_guard as G                                    # noqa: E402

PROTOCOL_MD = os.path.join(HERE, "PROTOCOL.md")

results = []

_TMPDIRS = []


def _temp_root(review_text):
    """A throwaway root containing a candidate ``ROOT_CAUSE_REVIEW.md``.

    Used instead of the real repository so the review can be tested present,
    draft, stale or absent without ever touching the real tree.
    """
    d = tempfile.mkdtemp(prefix="protocol_guard_")
    _TMPDIRS.append(d)
    os.makedirs(os.path.join(d, "experiments"), exist_ok=True)
    with open(os.path.join(d, "experiments", G.REVIEW_FILE), "w",
              encoding="utf-8") as fh:
        fh.write(review_text)
    return d


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print("  PASS  %s" % name)
    except AssertionError as e:
        results.append((name, False, str(e)))
        print("  FAIL  %s -> %s" % (name, e))
    except Exception as e:                                     # noqa: BLE001
        results.append((name, False, "%s: %s" % (type(e).__name__, e)))
        print("  ERROR %s -> %s: %s" % (name, type(e).__name__, e))


def synth(exp, verdict):
    """A history row for a hypothetical experiment with no results.json yet.

    No such directory exists, so ``own_gate`` returns None and the row is
    classified from its CSV verdict alone - which is exactly the situation of
    a brand new experiment.  Using synthetic rows rather than the real history
    is what lets every protocol test below survive Exp11 landing.
    """
    return G.Row(exp, exp + "_synthetic", verdict,
                 "experiments/%s_synthetic/RESULTS.md" % exp, 0)


def history(*verdicts):
    """A synthetic history: a PROMOTE, then one verdict per further argument.

    Deliberately not named after any real experiment, so a reader cannot mistake
    it for the recorded state of this repository.
    """
    rows = [G.Row("expA", "synthetic_promote", "promoting",
                  "experiments/expA/RESULTS.md", 1)]
    for i, verdict in enumerate(verdicts, start=2):
        tag = "exp%s" % chr(ord("A") + i - 1)
        rows.append(synth(tag, verdict))
    return rows


# ---------------------------------------------------------------------------
# 1. the threshold - the one number the whole stall rule turns on
# ---------------------------------------------------------------------------

def test_threshold_is_three_and_the_document_agrees():
    """Protocol text and code must state the same threshold, or agents will
    follow whichever they read first."""
    assert G.STALL_THRESHOLD == 3, G.STALL_THRESHOLD
    with open(PROTOCOL_MD, encoding="utf-8") as fh:
        doc = fh.read()
    assert "последние 3 последовательных" in doc, \
        "PROTOCOL.md does not state the 3-experiment stall rule verbatim"
    assert str(G.STALL_THRESHOLD) in doc, \
        "PROTOCOL.md does not mention the guard's threshold value"


def test_doc_records_the_required_review_questions():
    """All five review keys are named in the protocol, not just in the code."""
    with open(PROTOCOL_MD, encoding="utf-8") as fh:
        doc = fh.read()
    for key in G.REVIEW_KEYS:
        assert key in doc, "PROTOCOL.md never mentions the review key %r" % key


# ---------------------------------------------------------------------------
# 2. the two rules the guard exists to make structural
# ---------------------------------------------------------------------------

def test_failed_experiment_can_never_become_the_baseline():
    """``current_best`` must return a PROMOTE row even when STOPs sit after it."""
    rows = history("not_promoting", "not_promoting")
    best = G.current_best(rows)
    assert best is not None and best.exp == "expA", best
    # and a tail of STOPs must not drag the baseline along
    rows.append(synth("expD", "not_promoting"))
    assert G.current_best(rows).exp == "expA", \
        "a failed experiment was accepted as the new baseline"
    # the same holds whatever the failed experiment was called
    for name in ("exp11", "exp42", "exp99"):
        alt = history("not_promoting", "not_promoting")
        alt.append(synth(name, "not_promoting"))
        assert G.current_best(alt).exp == "expA", name


def test_last_promotion_becomes_the_current_best():
    """The baseline follows the most recent PROMOTE, wherever it sits.

    This is the property that keeps the protocol alive across a promotion: a
    new PROMOTE must become the baseline with no code and no doc edit.
    """
    rows = history("promoting", "not_promoting", "promoting", "not_promoting")
    best = G.current_best(rows)
    assert best.exp == "expD", best          # the later PROMOTE, not the first
    rep = G.check(rows=rows)
    assert rep["current_best"] == "expD", rep["current_best"]
    # and a promotion followed by nothing is still the best
    assert G.current_best(history("promoting", "promoting")).exp == "expC"


def test_two_stops_permit_the_next_experiment_and_three_block_it():
    """The stall boundary, as an executable property rather than a fixed date.

    Below the threshold a new experiment may start; at the threshold it may not
    be invented without a root-cause review.
    """
    base = history("not_promoting", "not_promoting")          # 2 STOPs

    r = G.check(rows=base)
    assert r["stall"]["n_consecutive_stops"] == 2, r["stall"]
    assert r["stall"]["blocking"] is False
    assert r["can_start_next_experiment"] is True, \
        "2 consecutive STOPS are below the threshold and must not block"
    assert r["current_best"] == "expA"

    # a third STOP tips it over - and must block, since no review exists
    r3 = G.check(rows=base + [synth("expD", "not_promoting")])
    assert r3["stall"]["n_consecutive_stops"] == 3, r3["stall"]
    assert r3["stall"]["blocking"] is True, \
        "%d consecutive STOPS must block the next experiment" % \
        G.STALL_THRESHOLD
    assert r3["can_start_next_experiment"] is False
    assert any("STALLED HILLCLIMB" in b for b in r3["blocking"]), r3["blocking"]
    # a stall blocks the next experiment but never demotes the baseline
    assert r3["current_best"] == "expA"
    # and the blocked report names the exact experiments the review must cover
    assert r3["stall"]["stopped"] == ["expB", "expC", "expD"], r3["stall"]


def test_a_promotion_resets_the_stall_counter():
    """A PROMOTE breaks the tail; the count is about consecutive STOPS."""
    rows = history("not_promoting", "not_promoting", "promoting",
                   "not_promoting")
    state = G.stall_state(rows)
    assert state["n_consecutive_stops"] == 1, state
    assert state["stopped"] == ["expE"], state
    assert G.current_best(rows).exp == "expD"
    # and a full reset: the stall is gone entirely
    assert G.stall_state(history("not_promoting", "promoting")) \
        ["n_consecutive_stops"] == 0


def test_packaged_rows_do_not_count_as_hillclimb_steps():
    """``sub*`` rows are builds of an already-promoted system, not hypotheses.

    Counting one would both inflate the experiment list and reset a stall, so
    the rule is pinned against the real history - a packaged build exists there
    and will keep existing, whatever experiment numbers come next.
    """
    rows = G.read_history()
    exps = [r.exp for r in G.experiments(rows)]
    assert not [e for e in exps if e.startswith("sub")], \
        "a packaged build was counted as a research experiment"
    # and a packaged row must not break a stall that precedes it
    with_build = G.check(rows=history("not_promoting", "not_promoting")
                         + [G.Row("sub99", "submission_x", "packaged",
                                  "submission_exp99/RESULTS.md", 9)])
    assert with_build["stall"]["n_consecutive_stops"] == 2, with_build["stall"]
    assert with_build["can_start_next_experiment"] is True


# ---------------------------------------------------------------------------
# 3. the two sources of truth must agree
# ---------------------------------------------------------------------------

def test_history_and_own_gate_agree_on_every_experiment():
    """``results.csv`` verdict and each run's ``promotion_gate`` must agree.

    A disagreement is the failure mode this rule exists for: a row that claims
    STOP while its own results.json says the gate PASSED would otherwise be
    miscounted, and a false promotion would silently unblock a stalled climb.
    """
    rows = G.read_history()
    for row in G.experiments(rows):
        kind, reason = G.classify(row, ROOT)
        assert kind != "conflict", "%s: %s" % (row.exp, reason)


def test_a_disagreement_is_fail_safe():
    """An unrecognised or missing verdict must never be read as a promotion."""
    row = synth("exp99", "not_promoting")
    assert G.classify(row)[0] == "stop"          # no results.json -> CSV only
    assert G.classify(synth("exp99", "promoting"))[0] == "promote"

    # an unknown verdict must never be read as a promotion
    assert G.classify(synth("exp99", "great_new_thing"))[0] == "stop"


def test_csv_versus_own_gate_drift_is_reported_and_blocks():
    """Drift between the history and a run's own results.json must be loud.

    Built in a temp root from synthetic rows, so it does not depend on which
    experiment happens to sit in the history when this runs.
    """
    root = _temp_root("")
    stop_row = synth("exp50", "not_promoting")
    assert G.own_gate(stop_row, root) is None, \
        "the synthetic row was expected to have no results.json"

    def with_gate(tag, verdict, passed):
        d = os.path.join(root, "experiments", "%s_synthetic" % tag)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as fh:
            json.dump({"promotion_gate": {"passed": passed}}, fh)
        return synth(tag, verdict)

    # CSV says STOP, the run's own gate says PASS
    drifted_stop = with_gate("exp50", "not_promoting", True)
    assert G.own_gate(drifted_stop, root) is True
    assert G.classify(drifted_stop, root)[0] == "conflict", \
        "CSV STOP vs own gate PASS was not reported as a conflict"

    # the mirror image: CSV says PROMOTE, the run's own gate says FAIL
    drifted_promote = with_gate("exp51", "promoting", False)
    assert G.classify(drifted_promote, root)[0] == "conflict", \
        "CSV PROMOTE vs own gate FAIL was not reported as a conflict"

    # a conflict must block a new experiment, not merely be reported
    rep = G.check(rows=history("not_promoting", "not_promoting")
                  + [drifted_stop, drifted_promote], root=root)
    assert rep["can_start_next_experiment"] is False, \
        "a contradictory history did not block the next experiment"
    assert any("exp50" in v for v in rep["violations"]), rep["violations"]
    assert any("exp51" in v for v in rep["violations"]), rep["violations"]


# ---------------------------------------------------------------------------
# 4. the root-cause review - a file, and it must be current
# ---------------------------------------------------------------------------

def stalled_tail():
    """A synthetic 3-STOP tail, so review tests never depend on real numbers."""
    rows = history("not_promoting", "not_promoting", "not_promoting")
    tail = G.consecutive_stops(rows)
    assert len(tail) == G.STALL_THRESHOLD, tail
    return rows, tail


def review_text(covers, status="complete"):
    """A review header covering exactly ``covers``."""
    return ("status: %s\ncovers: [%s]\n"
            "hypotheses: a\ntransfer: b\nheadroom: c\nnoise: d\nwarrant: e\n"
            % (status, ", ".join(covers)))


def test_no_review_means_stall_blocks_and_names_the_missing_pieces():
    """The guard must block, and must tell the agent what the review must contain.

    What the agent actually reads is the CLI output, so that is what is
    asserted - including the five questions, which is the difference between a
    guard that blocks and a guard that merely refuses.
    """
    rows, _ = stalled_tail()
    rep = G.check(rows=rows)
    assert rep["stall"]["blocking"] is True
    assert rep["can_start_next_experiment"] is False
    assert any(G.REVIEW_FILE in r for r in rep["block_reasons"]), rep["block_reasons"]

    out = G._fmt(rep)
    assert G.REVIEW_FILE in out, out
    # all five questions AND the covers header must be spelled out, because the
    # covers list is what tells the agent WHICH experiments the review is for
    for key in G.REVIEW_QUESTION_KEYS:
        assert key in out, "the guard did not name the missing key %r" % key
    assert "covers" in out, "the guard did not say the review must declare covers"
    assert "status: complete" in out, out
    assert "DO NOT invent the next experiment" in out, out


def test_a_current_review_unblocks_the_stall():
    """A complete review of exactly the stalled experiments lets work resume."""
    rows, tail = stalled_tail()
    covers = [r.exp for r in tail]
    rep = G.check(rows=rows,
                  root=_temp_root(review_text(covers)))
    assert rep["stall"]["review_ok"] is True, rep["stall"]
    assert rep["can_start_next_experiment"] is True, \
        "a complete, current review did not unblock the stall"
    # blocking stays true - the stall happened - but it no longer forbids work
    assert rep["stall"]["blocking"] is True


def test_a_stale_review_does_not_unblock_a_new_stall():
    """A review of the PREVIOUS tail must not satisfy a NEW stall.

    This is the case that matters: once a third STOP lands, a review written
    about the earlier pair would otherwise clear the guard forever.
    """
    rows, tail = stalled_tail()
    covers = [r.exp for r in tail]
    stale = covers[:-1]

    root = _temp_root(review_text(stale))
    defects = G.review_defects(tail, root=root)
    assert defects, "a review covering only %r was accepted for a %d-experiment " \
                    "stall" % (stale, len(covers))
    assert any("covers" in d for d in defects), defects

    # and end to end: the guard must still refuse
    rep = G.check(rows=rows, root=root)
    assert rep["can_start_next_experiment"] is False, \
        "a stale review unblocked a new stall"


def test_review_header_must_be_complete_and_name_every_stalled_experiment():
    """A draft, or a review missing a key, is not a review."""
    _, tail = stalled_tail()
    covers = [r.exp for r in tail]
    good = review_text(covers)

    meta = G.parse_review(good)
    assert meta["status"] == "complete" and meta["hypotheses"] == "a"

    def defects_for(text):
        return G.review_defects(tail, root=_temp_root(text))

    assert not defects_for(good), \
        "a complete, current review was rejected: %r" % defects_for(good)
    assert defects_for(review_text(covers, status="draft")), \
        "a draft review was accepted"
    assert defects_for(good.replace("warrant: e", "")), \
        "a review missing the 'warrant' key was accepted"
    assert defects_for(review_text(covers[:-1])), \
        "a review missing the newest stalled experiment was accepted"
    # a review file that is not there at all
    assert G.review_defects(tail, root=os.path.join(ROOT, "does_not_exist"))


def test_a_review_for_a_superseded_tail_is_rejected():
    """Once a PROMOTE resets the tail, a review of the old tail is moot.

    The inverse of the stale case: a review written for the three STOPs must
    stop being consulted the moment one of them is promoted and the tail
    empties.  ``review_defects`` is the right level to assert on - whether the
    guard happens to consult the file for a non-stalled history is a detail of
    ``check``, and asserting it would re-couple the test to that.
    """
    _, tail = stalled_tail()
    old = [r.exp for r in tail]
    rows = history("not_promoting", "not_promoting", "not_promoting",
                   "promoting")
    new_tail = G.consecutive_stops(rows)
    assert new_tail == [], new_tail
    assert G.current_best(rows).exp == "expE", \
        "the promoted experiment did not become the new baseline"

    # a review still claiming that old tail no longer describes this history
    defects = G.review_defects(new_tail, root=_temp_root(review_text(old)))
    assert defects, \
        "a review of a superseded tail was still accepted"
    assert any("covers" in d for d in defects), defects

    # the same review remains valid for the tail it was actually written about
    assert G.review_defects(tail, root=_temp_root(review_text(old))) == []


# ---------------------------------------------------------------------------
# 5. the guard must not touch anything
# ---------------------------------------------------------------------------

def test_running_the_guard_does_not_modify_the_history():
    """Read-only is the whole point: the guard is an advisor, not a writer."""
    # the history, the protocol, and every results.json the guard reads -
    # collected from the history itself so the set follows Exp11 onwards
    files = [G.RESULTS_CSV, PROTOCOL_MD]
    for row in G.read_history():
        path = os.path.join(G.exp_dir(row, ROOT), "results.json")
        if os.path.exists(path):
            files.append(path)
    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
              for p in files}
    G.check()
    G.main()
    after = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
             for p in files}
    assert before == after, "the guard modified %r" % [
        p for p in files if before[p] != after[p]]


# ---------------------------------------------------------------------------
# 6. the real history must satisfy the same properties
# ---------------------------------------------------------------------------

def test_the_real_history_is_internally_consistent():
    """Whatever the recorded state is, the guard must accept it without drift.

    Written as properties rather than as an expected snapshot, so it keeps
    holding after Exp11 lands, after a promotion, and after a root-cause review
    is written.  Only genuine inconsistency fails it.
    """
    rep = G.check()
    assert rep["current_best"] is not None, "no promoted experiment recorded"
    assert rep["current_best"] in rep["experiments"], rep
    assert rep["stall"]["n_consecutive_stops"] == len(rep["stall"]["stopped"])
    assert rep["stall"]["n_consecutive_stops"] < rep["stall"]["threshold"] or \
        rep["stall"]["review_ok"], \
        "the history is stalled with no current review, yet nothing reports it"
    # a violation may be reported, but it must never be silently ignored
    if rep["violations"]:
        assert not rep["can_start_next_experiment"], \
            "violations were reported but the guard still allowed an experiment"
    # and the current best must be a real, promoted row on disk
    best = G.current_best(G.read_history(), ROOT)
    assert G.own_gate(best, ROOT) in (None, True), \
        "the current best is not a promoted experiment"


def test_failed_experiments_left_no_production_build():
    """Rule 6, checked against the filesystem rather than trusted.

    Derived from the history, so it covers every failed experiment present now
    and any added later.
    """
    checked = 0
    for row in G.experiments(G.read_history()):
        if G.own_gate(row, ROOT) is False:
            assert G.stray_builds(row, ROOT) == [], \
                "stray build for the failed experiment %s" % row.exp
            checked += 1
    # a history with no recorded gate failures has nothing to check; that is a
    # valid state, not a failure of this test
    assert checked >= 0, checked


def test_results_csv_schema_is_what_the_guard_expects():
    """The column indices are hard-coded; a schema change must be loud.

    The row COUNT is deliberately not asserted - it grows with every experiment.
    What matters is that every row the guard reads is well formed and points at
    a report that exists.
    """
    rows = G.read_history()
    assert rows, "results.csv parsed to nothing"
    for row in rows:
        assert row.exp and row.verdict, row
        assert row.results_rel.endswith("RESULTS.md"), row.results_rel
        assert os.path.exists(os.path.join(ROOT, row.results_rel.replace(
            "/", os.sep))), "results.csv points at a missing report: %s" \
            % row.results_rel
    # a truncated line must raise, not be silently skipped
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8", newline="") as fh:
        fh.write("exp01,name\n")
        bad = fh.name
    try:
        G.read_history(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a short results.csv line was accepted")
    finally:
        os.unlink(bad)


def main():
    print("=" * 72)
    print("Experiment protocol guard tests")
    print("=" * 72)
    tests = [
        ("stall threshold is 3 and PROTOCOL.md says 3",
         test_threshold_is_three_and_the_document_agrees),
        ("PROTOCOL.md names all five review questions",
         test_doc_records_the_required_review_questions),
        ("a failed experiment can never become the baseline",
         test_failed_experiment_can_never_become_the_baseline),
        ("the last promotion becomes the current best",
         test_last_promotion_becomes_the_current_best),
        ("2 STOPs permit the next experiment; 3 block it",
         test_two_stops_permit_the_next_experiment_and_three_block_it),
        ("a promotion resets the stall counter",
         test_a_promotion_resets_the_stall_counter),
        ("packaged rows are not hillclimb steps",
         test_packaged_rows_do_not_count_as_hillclimb_steps),
        ("results.csv and each run's own gate agree",
         test_history_and_own_gate_agree_on_every_experiment),
        ("an unknown verdict is never read as a promotion",
         test_a_disagreement_is_fail_safe),
        ("csv/own-gate drift is reported and blocks",
         test_csv_versus_own_gate_drift_is_reported_and_blocks),
        ("a stall with no review blocks and names what is missing",
         test_no_review_means_stall_blocks_and_names_the_missing_pieces),
        ("a current review unblocks the stall",
         test_a_current_review_unblocks_the_stall),
        ("a stale review does not unblock a new stall",
         test_a_stale_review_does_not_unblock_a_new_stall),
        ("a review must be complete and current to count",
         test_review_header_must_be_complete_and_name_every_stalled_experiment),
        ("a review for a superseded tail is no longer consulted",
         test_a_review_for_a_superseded_tail_is_rejected),
        ("running the guard modifies nothing",
         test_running_the_guard_does_not_modify_the_history),
        ("the real history is internally consistent",
         test_the_real_history_is_internally_consistent),
        ("failed experiments left no production build",
         test_failed_experiments_left_no_production_build),
        ("results.csv schema is what the guard expects",
         test_results_csv_schema_is_what_the_guard_expects),
    ]
    for name, fn in tests:
        check(name, fn)

    npass = sum(1 for _, o, _ in results if o)
    print("\n%d/%d passed" % (npass, len(results)))
    for name, o, err in results:
        if not o:
            print("  FAILED: %s -> %s" % (name, err))
    for d in _TMPDIRS:
        for dirpath, _, files in os.walk(d, topdown=False):
            for f in files:
                os.unlink(os.path.join(dirpath, f))
            os.rmdir(dirpath)
        if os.path.isdir(d):
            os.rmdir(d)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
