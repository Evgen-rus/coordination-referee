"""Tests for the experiment protocol guard.

Deliberately few.  A test that cannot fail on a real regression is worse than
no test, so each case here pins a rule that the protocol depends on and that a
future edit could plausibly break:

  * the stall threshold is 3, and PROTOCOL.md says 3 (doc/code drift guard);
  * a failed experiment can never become the baseline;
  * ``results.csv`` and the run's own ``promotion_gate`` must agree, and a
    disagreement is fail-safe - it counts as STOP, never as a promotion;
  * a stale root-cause review must not unblock a new stall;
  * the guard is READ-ONLY: running it must not touch the history;
  * the two STOPS recorded today are real STOPS, and Exp11 is the third.

Run:  python experiments/test_protocol_guard.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import protocol_guard as G                                    # noqa: E402

PROTOCOL_MD = os.path.join(HERE, "PROTOCOL.md")
REVIEW_MD = os.path.join(HERE, G.REVIEW_FILE)

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
    a brand new experiment.
    """
    return G.Row(exp, exp + "_synthetic", verdict,
                 "experiments/%s_synthetic/RESULTS.md" % exp, 0)


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
    """``current_best`` must return a PROMOTE row even when a STOP sits after it."""
    rows = [synth("exp08", "promoting"), synth("exp09", "not_promoting"),
            synth("exp10", "not_promoting")]
    best = G.current_best(rows)
    assert best is not None and best.exp == "exp08", best
    # and a tail of STOPs must not drag the baseline along
    rows.append(synth("exp11", "not_promoting"))
    assert G.current_best(rows).exp == "exp08", \
        "a failed experiment was accepted as the new baseline"


def test_two_stops_permit_the_third_experiment_and_three_block_it():
    """The rule the user asked for, as an executable boundary.

    2 consecutive STOPS -> Exp11 may run.  3 -> Exp12 may not be invented.
    """
    base = [synth("exp08", "promoting"), synth("exp09", "not_promoting"),
            synth("exp10", "not_promoting")]

    r = G.check(rows=base)
    assert r["current_best"] == "exp08", r["current_best"]
    assert r["stall"]["n_consecutive_stops"] == 2
    assert r["stall"]["stopped"] == ["exp09", "exp10"]
    assert r["stall"]["blocking"] is False
    assert r["can_start_next_experiment"] is True, \
        "Exp11 must be permitted: it is the third experiment of this cycle"

    # a third STOP tips it over - and must block, since no review exists yet
    r3 = G.check(rows=base + [synth("exp11", "not_promoting")])
    assert r3["stall"]["n_consecutive_stops"] == 3
    assert r3["stall"]["blocking"] is True, \
        "3 consecutive STOPS must block the next experiment"
    assert r3["can_start_next_experiment"] is False
    assert any("STALLED HILLCLIMB" in b for b in r3["blocking"]), r3["blocking"]
    # and the baseline must still be exp08 - a stall does not demote it
    assert r3["current_best"] == "exp08"


def test_a_promotion_resets_the_stall_counter():
    """A PROMOTE breaks the tail; the count is about consecutive STOPS."""
    rows = [synth("exp08", "promoting"), synth("exp09", "not_promoting"),
            synth("exp10", "not_promoting"), synth("exp11", "promoting"),
            synth("exp12", "not_promoting")]
    state = G.stall_state(rows)
    assert state["n_consecutive_stops"] == 1, state
    assert state["stopped"] == ["exp12"]
    assert G.current_best(rows).exp == "exp11"


def test_packaged_rows_do_not_count_as_hillclimb_steps():
    """``sub08`` is a build of an already-promoted system, not a hypothesis.

    Counting it would both inflate the experiment list and reset a stall, so
    the rule is pinned here rather than left to the reader of the code.
    """
    rows = G.read_history()
    exps = [r.exp for r in G.experiments(rows)]
    assert not [e for e in exps if e.startswith("sub")], \
        "a packaged build was counted as a research experiment"
    # the real history is 11 research runs and its packaged rows are excluded
    assert len(exps) == 11, exps


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
    """Construct the drift and prove it resolves to STOP, not to a promotion."""
    row = synth("exp99", "not_promoting")
    assert G.classify(row)[0] == "stop"          # no results.json -> CSV only
    assert G.classify(synth("exp99", "promoting"))[0] == "promote"

    # an unknown verdict must never be read as a promotion
    assert G.classify(synth("exp99", "great_new_thing"))[0] == "stop"

    # real drift, using exp09 whose results.json says passed=False
    real = [r for r in G.read_history() if r.exp == "exp09"][0]
    assert G.own_gate(real, ROOT) is False
    bad = G.Row("exp09", real.name, "promising", real.results_rel, 1)
    kind, reason = G.classify(bad, ROOT)
    assert kind == "conflict", (kind, reason)
    # a conflict must block a new experiment, not merely be reported
    rep = G.check(rows=rows_have_conflict())
    assert rep["can_start_next_experiment"] is False, \
        "a contradictory history row did not block the next experiment"
    assert any("exp09" in v for v in rep["violations"]), rep["violations"]


def rows_have_conflict():
    rows = G.read_history()
    real = [r for r in rows if r.exp == "exp09"][0]
    rows[-1] = G.Row("exp09", real.name, "promising", real.results_rel, 1)
    return rows


# ---------------------------------------------------------------------------
# 4. the root-cause review - a file, and it must be current
# ---------------------------------------------------------------------------

def test_no_review_means_stall_blocks_and_names_the_missing_pieces():
    """The guard must block, and must tell the agent what the review must contain.

    What the agent actually reads is the CLI output, so that is what is
    asserted - including the five questions, which is the difference between a
    guard that blocks and a guard that merely refuses.
    """
    rows = G.read_history() + [synth("exp11", "not_promoting")]
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


def test_a_stale_review_does_not_unblock_a_new_stall():
    """A review of the OLD tail must not satisfy a NEW stall.

    This is the case that matters: after Exp11 STOPS, a review written about
    exp09/exp10 alone would otherwise clear the guard forever.
    """
    tail = G.consecutive_stops(G.read_history() + [synth("exp11", "not_promoting")])
    assert [r.exp for r in tail] == ["exp09", "exp10", "exp11"]

    # a complete review of the PREVIOUS tail only
    root = _temp_root("status: complete\ncovers: [exp09, exp10]\n"
                      "hypotheses: a\ntransfer: b\nheadroom: c\nnoise: d\n"
                      "warrant: e\n")
    defects = G.review_defects(tail, root=root)
    assert defects, "a review covering only exp09+exp10 was accepted for a " \
                    "three-experiment stall"
    assert any("covers" in d for d in defects), defects

    # and the same stale review must leave the guard blocking
    rep = G.check(rows=G.read_history() + [synth("exp11", "not_promoting")])
    assert rep["can_start_next_experiment"] is False


def test_review_header_must_be_complete_and_name_every_stalled_experiment():
    """A draft, or a review missing a key, is not a review."""
    good = ("status: complete\ncovers: [exp09, exp10, exp11]\n"
            "hypotheses: a\ntransfer: b\nheadroom: c\nnoise: d\nwarrant: e\n")
    meta = G.parse_review(good)
    assert meta["status"] == "complete" and meta["hypotheses"] == "a"

    tail = G.consecutive_stops(G.read_history() + [synth("exp11", "not_promoting")])

    def defects_for(text):
        return G.review_defects(tail, root=_temp_root(text))

    assert not defects_for(good), \
        "a complete, current review was rejected: %r" % defects_for(good)
    assert defects_for(good.replace("status: complete", "status: draft"))
    assert defects_for(good.replace("warrant: e", ""))
    assert defects_for(good.replace("covers: [exp09, exp10, exp11]",
                                    "covers: [exp09, exp10]"))
    # a review file that is not there at all
    assert G.review_defects(tail, root=os.path.join(ROOT, "does_not_exist"))


def test_no_review_is_committed_and_none_is_invented_for_exp11():
    """There is no review on disk, and that is deliberate.

    Exp11 does not exist yet, so filling in a root-cause review now would mean
    inventing its result.  The guard is expected to report 'absent'.
    """
    assert not os.path.exists(REVIEW_MD), \
        "%s exists - a review must be written for a stall that has happened, " \
        "not before" % G.REVIEW_FILE
    state = G.stall_state()
    assert state["review_ok"] is False
    assert state["blocking"] is False, \
        "2 STOPS is below the threshold, so the absent review must not block"


# ---------------------------------------------------------------------------
# 5. the guard must not touch anything
# ---------------------------------------------------------------------------

def test_running_the_guard_does_not_modify_the_history():
    """Read-only is the whole point: the guard is an advisor, not a writer."""
    files = [G.RESULTS_CSV, PROTOCOL_MD,
             os.path.join(ROOT, "experiments", "exp09_decision_calibration",
                          "results.json"),
             os.path.join(ROOT, "experiments", "exp10_ab_probability_blend",
                          "results.json")]
    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
              for p in files}
    G.check()
    G.main()
    after = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
             for p in files}
    assert before == after, "the guard modified %r" % [
        p for p in files if before[p] != after[p]]


# ---------------------------------------------------------------------------
# 6. the recorded history this guard depends on
# ---------------------------------------------------------------------------

def test_current_best_is_exp08_and_the_two_stops_are_real():
    """Pin the state the protocol was written against.

    If a later PROMOTE lands, this test is EXPECTED to fail - that is the
    signal to re-read PROTOCOL.md section 4, not a bug.
    """
    rep = G.check()
    assert rep["current_best"] == "exp08", rep["current_best"]
    assert rep["stall"]["n_consecutive_stops"] == 2, rep["stall"]
    assert rep["stall"]["stopped"] == ["exp09", "exp10"], rep["stall"]
    assert rep["violations"] == [], rep["violations"]
    assert rep["can_start_next_experiment"] is True


def test_failed_experiments_left_no_production_build():
    """Rule 6, checked against the filesystem rather than trusted.

    Exp09 and Exp10 both failed their gate; neither may have a submission
    directory or a ZIP.
    """
    for name in ("submission_exp09", "submission_exp10",
                 "submission_exp09_decision_calibration.zip",
                 "submission_exp10_ab_probability_blend.zip"):
        assert not os.path.exists(os.path.join(ROOT, name)), \
            "%s exists although its experiment STOPped" % name
    for row in G.experiments(G.read_history()):
        if G.own_gate(row, ROOT) is False:
            assert G.stray_builds(row, ROOT) == [], \
                "stray build for the failed experiment %s" % row.exp


def test_results_csv_schema_is_what_the_guard_expects():
    """The column indices are hard-coded; a schema change must be loud."""
    rows = G.read_history()
    assert len(rows) == 14, len(rows)
    for row in rows:
        assert row.results_rel.endswith("RESULTS.md"), row.results_rel
        assert os.path.exists(os.path.join(ROOT, row.results_rel.replace(
            "/", os.sep))), "results.csv points at a missing report: %s" \
            % row.results_rel
    # a truncated line must raise, not be silently skipped
    import tempfile
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
        ("2 STOPs permit Exp11; 3 STOPs block Exp12",
         test_two_stops_permit_the_third_experiment_and_three_block_it),
        ("a promotion resets the stall counter",
         test_a_promotion_resets_the_stall_counter),
        ("packaged rows are not hillclimb steps",
         test_packaged_rows_do_not_count_as_hillclimb_steps),
        ("results.csv and each run's own gate agree",
         test_history_and_own_gate_agree_on_every_experiment),
        ("a disagreement is fail-safe and blocks",
         test_a_disagreement_is_fail_safe),
        ("a stall with no review blocks and names what is missing",
         test_no_review_means_stall_blocks_and_names_the_missing_pieces),
        ("a stale review does not unblock a new stall",
         test_a_stale_review_does_not_unblock_a_new_stall),
        ("a review must be complete and current to count",
         test_review_header_must_be_complete_and_name_every_stalled_experiment),
        ("no review is committed, and none invented for Exp11",
         test_no_review_is_committed_and_none_is_invented_for_exp11),
        ("running the guard modifies nothing",
         test_running_the_guard_does_not_modify_the_history),
        ("current best is exp08 and the two STOPS are real",
         test_current_best_is_exp08_and_the_two_stops_are_real),
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
