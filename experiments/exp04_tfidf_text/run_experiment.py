"""Experiment 04 driver: light TF-IDF text signal blended with the Exp03 B tabular model.

Systems compared on the SAME 3 folds used by every previous experiment
(``StratifiedKFold(3, shuffle=True, random_state=0)``):

  1. baseline_flat   122 official features, LightGBM
  2. exp03_B         122 + 63 normalised, LightGBM        (tabular foundation)
  3. text_word       word TF-IDF (1-2 grams)  + LogisticRegression
  4. text_char       char_wb TF-IDF (3-5)    + LogisticRegression
  5. text_wc         word+char hstack        + LogisticRegression
  6. blend           alpha * P_tabular + (1 - alpha) * P_text

alpha and the text variant are chosen by an INNER split of the training fold
only (never the validation fold).  Fixed alphas 0.25/0.50/0.75 are also
reported.  Success F1 is the saved Exp03 B OOF (held fixed, as instructed) and
fault_turn uses the official rule-based localiser on the predicted label, so
composite stays honest.

Run:  .\\.venv\\Scripts\\python.exe experiments\\exp04_tfidf_text\\run_experiment.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(HERE, "..", "exp03_length_normalization"))

import mem                                              # noqa: E402
from features import extract_features, parse_run       # noqa: E402
from localize import localize                          # noqa: E402
from metrics import binary_f1, composite, f1_per_class  # noqa: E402
from metrics import fault_turn_hit_at_k, macro_f1      # noqa: E402
import new_features as nf                              # noqa: E402
from build_docs import domain_proxy                     # noqa: E402

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
L2I = {l: i for i, l in enumerate(LABELS)}
ALPHAS = [0.25, 0.50, 0.75]
INNER_SEED = 7
TEXT_TAGS = ("text_word", "text_char", "text_wc")

_T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - _T0, m), flush=True)


def mf1(t, p):
    assert len(t) == len(p)
    return macro_f1(list(t), list(p))


def lgb_label():
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=7, n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)


def logreg():
    """Linear model for sparse high-dimensional text.  lbfgs (sklearn default)
    is stable for this 7-class problem at this matrix size; not tuned."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=4.0, solver="lbfgs", tol=1e-4)


def word_vec():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=5,
                           sublinear_tf=True, max_features=60000,
                           token_pattern=r"\S+", dtype=np.float32)


def char_vec():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           sublinear_tf=True, max_features=60000,
                           lowercase=False, dtype=np.float32)


def fit_text(tag, tr_docs, tr_y, va_docs):
    """Fit a TF-IDF + LogisticRegression system; return (proba, meta)."""
    t0 = time.time()
    vw, vc = (word_vec(), None) if tag == "text_word" else \
             (None, char_vec()) if tag == "text_char" else (word_vec(), char_vec())
    A = vw.fit_transform(tr_docs) if vw is not None else None
    B = vc.fit_transform(tr_docs) if vc is not None else None
    Xtr = sparse.hstack([m for m in (A, B) if m is not None], format="csr") \
        if (A is not None and B is not None) else (A if A is not None else B)
    t1 = time.time()
    clf = logreg().fit(Xtr, tr_y)
    t2 = time.time()
    Av = vw.transform(va_docs) if vw is not None else None
    Bv = vc.transform(va_docs) if vc is not None else None
    Xva = sparse.hstack([m for m in (Av, Bv) if m is not None], format="csr") \
        if (Av is not None and Bv is not None) else (Av if Av is not None else Bv)
    P = np.zeros((Xva.shape[0], 7))
    P[:, clf.classes_] = clf.predict_proba(Xva)
    t3 = time.time()
    if vw is not None and vc is not None:
        names_out = (np.asarray(vw.get_feature_names_out()),
                     np.asarray(vc.get_feature_names_out()))
    elif vw is not None:
        names_out = np.asarray(vw.get_feature_names_out())
    else:
        names_out = np.asarray(vc.get_feature_names_out())
    meta = {"n_features": int(Xtr.shape[1]), "nnz": int(Xtr.nnz),
            "rows": int(Xtr.shape[0]),
            "bytes": int(Xtr.data.nbytes + Xtr.indices.nbytes + Xtr.indptr.nbytes),
            "fit_s": t2 - t0, "infer_s": t3 - t2,
            "names": names_out}
    coefs = clf.coef_.copy()
    del A, B, Av, Bv, Xtr, Xva, clf, vw, vc
    gc.collect()
    return P, meta, coefs


def main():
    from sklearn.model_selection import StratifiedKFold, train_test_split

    train = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    yi = train["label"].map(L2I).values
    ys = np.array([LABELS[i] for i in yi], dtype=object)
    s = train["success"].values
    ft = train["fault_turn"].values
    n = len(yi)

    # ---------------- tabular: Exp03 B construction, reproduced exactly ----------------
    log("parsing runs + official 122 features")
    runs = [parse_run(r) for r in train.to_dict("records")]
    base_rows = [extract_features(r) for r in runs]
    Xb = pd.DataFrame(base_rows).fillna(0.0)
    t0 = time.time()
    Xn = nf.build_matrix(runs, base_rows)
    log("exp03 B tabular: %d cols (features %.1fs)" % (Xb.shape[1] + Xn.shape[1], time.time() - t0))
    XA = Xb.copy()
    XB = pd.concat([Xb, Xn], axis=1)
    nmsg = Xb["n_messages"].values
    del base_rows
    gc.collect()

    # ---------------- text documents ----------------
    docs_path = os.path.join(HERE, "docs_train.json")
    if os.path.exists(docs_path):
        with open(docs_path, encoding="utf-8") as f:
            docs = json.load(f)
        log("loaded %d cached documents" % len(docs))
    else:
        from build_docs import build_all
        docs, _d, _ = build_all(train)
        log("built %d documents" % len(docs))
    doms = np.array([domain_proxy(g) for g in train["goal"].values])

    # ---------------- Exp03 B OOF success (held fixed) ----------------
    exp03 = pd.read_csv(os.path.join(HERE, "..", "exp03_length_normalization",
                                     "oof_B_baseline_plus_norm.csv"))
    assert (exp03["run_id"].values == train["run_id"].values).all()
    succ_b = exp03["success_pred"].values
    SUCCESS_F1 = float(binary_f1(list(s), list(succ_b)))
    log("Exp03 B Success F1 (fixed) = %.4f" % SUCCESS_F1)

    splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(XB, yi))
    systems = ["baseline_flat", "exp03_B"] + list(TEXT_TAGS)
    proba = {k: np.zeros((n, 7)) for k in ("exp03_B",) + TEXT_TAGS}
    fold_m = {k: [] for k in systems}
    meta = {k: [] for k in TEXT_TAGS}
    coefs = {k: [] for k in TEXT_TAGS}
    feat_names = {}
    blend_sel = []
    fixed_blend = {a: np.zeros(n, dtype=int) for a in ALPHAS}

    for k, (tr, va) in enumerate(splits):
        log("=" * 72)
        log("FOLD %d  train=%d  val=%d" % (k, len(tr), len(va)))
        tr_docs = [docs[i] for i in tr]
        va_docs = [docs[i] for i in va]

        # ---- 1) baseline flat ----
        mA = lgb_label().fit(XA.iloc[tr], yi[tr])
        pA = mA.predict(XA.iloc[va])
        fold_m["baseline_flat"].append(mf1([ys[i] for i in va], [LABELS[j] for j in pA]))
        log("  baseline_flat fold macro_f1=%.4f" % fold_m["baseline_flat"][-1])

        # ---- 2) Exp03 B ----
        mB = lgb_label().fit(XB.iloc[tr], yi[tr])
        pr = mB.predict_proba(XB.iloc[va])
        Pb = np.zeros((len(va), 7))
        Pb[:, mB.classes_] = pr
        proba["exp03_B"][va] = Pb
        pb = np.argmax(Pb, axis=1)
        fold_m["exp03_B"].append(mf1([ys[i] for i in va], [LABELS[j] for j in pb]))
        log("  exp03_B       fold macro_f1=%.4f" % fold_m["exp03_B"][-1])
        del mA, mB
        gc.collect()

        # ---- 3-5) text systems ----
        for tag in TEXT_TAGS:
            P, mt, cf = fit_text(tag, tr_docs, yi[tr], va_docs)
            proba[tag][va] = P
            pt = np.argmax(P, axis=1)
            fold_m[tag].append(mf1([ys[i] for i in va], [LABELS[j] for j in pt]))
            meta[tag].append(mt)
            coefs[tag].append(cf)
            feat_names.setdefault(tag, mt["names"])
            log("  %-10s fold macro_f1=%.4f  vocab=%6d  nnz=%8d  X=%.0f MB  fit=%.1fs  infer=%.2fs"
                % (tag, fold_m[tag][-1], mt["n_features"], mt["nnz"],
                   mt["bytes"] / 1e6, mt["fit_s"], mt["infer_s"]))

        # ---- honest alpha + text-variant choice: INNER split of the TRAIN fold ----
        itr, iva = train_test_split(tr, test_size=0.4, random_state=INNER_SEED,
                                    stratify=yi[tr])
        idocs = [docs[i] for i in itr]
        iya_docs = [docs[i] for i in iva]
        inner_p = {}
        for tag in TEXT_TAGS:
            Pi, _m, _c = fit_text(tag, idocs, yi[itr], iya_docs)
            inner_p[tag] = Pi
        mt_ = lgb_label().fit(XB.iloc[itr], yi[itr])
        prt = mt_.predict_proba(XB.iloc[iva])
        Pt = np.zeros((len(iva), 7))
        Pt[:, mt_.classes_] = prt
        inner_y = [LABELS[yi[i]] for i in iva]
        scores = {}
        for tag in TEXT_TAGS:
            for a in ALPHAS:
                mix = a * Pt + (1 - a) * inner_p[tag]
                scores[(tag, a)] = mf1(inner_y, [LABELS[i] for i in np.argmax(mix, axis=1)])
        top = max(scores, key=scores.get)
        blend_sel.append({"fold": k, "text": top[0], "alpha_tabular": top[1],
                          "inner_macro_f1": scores[top]})
        for a in ALPHAS:
            mix = a * proba["exp03_B"][va] + (1 - a) * proba[top[0]][va]
            fixed_blend[a][va] = np.argmax(mix, axis=1)
        log("  inner pick: text=%s alpha=%.2f (inner macro %.4f);  [%.2f]%.4f  [%.2f]%.4f  [%.2f]%.4f"
            % (top[0], top[1], scores[top],
               0.25, mf1([ys[i] for i in va], [LABELS[i] for i in fixed_blend[0.25][va]]),
               0.50, mf1([ys[i] for i in va], [LABELS[i] for i in fixed_blend[0.50][va]]),
               0.75, mf1([ys[i] for i in va], [LABELS[i] for i in fixed_blend[0.75][va]])))
        del mt_, Pt, inner_p
        gc.collect()

    # ---- honest blend: per-fold inner-selected (text, alpha) ----
    honest = np.zeros(n, dtype=int)
    for k, (tr, va) in enumerate(splits):
        tag = blend_sel[k]["text"]
        a = blend_sel[k]["alpha_tabular"]
        honest[va] = np.argmax(a * proba["exp03_B"][va] + (1 - a) * proba[tag][va], axis=1)

    all_pred = {k: np.argmax(proba[k], axis=1) for k in proba}
    log("refitting baseline_flat OOF predictions for a fair per-class table")
    oofA = np.zeros(n, dtype=int)
    for tr, va in splits:
        oofA[va] = lgb_label().fit(XA.iloc[tr], yi[tr]).predict(XA.iloc[va])
    all_pred["baseline_flat"] = oofA
    for a in ALPHAS:
        all_pred["blend_a%.2f" % a] = fixed_blend[a]
    all_pred["blend_honest"] = honest

    # ---------------- masks ----------------
    hard = (nmsg > np.median(nmsg)) & ((Xb["topo_mesh"].values > 0) |
                                       (Xb["topo_blackboard"].values > 0))
    no_intent = Xb["n_intents"].values <= 1
    compliance = doms == "compliance_audit"
    q66 = np.quantile(nmsg, 2 / 3)
    q80 = np.quantile(nmsg, 0.8)
    MASKS = {
        "no_intent": no_intent,
        "has_intent": ~no_intent,
        "long(>p66)": nmsg > q66,
        "longest_20%": nmsg >= q80,
        "hard/robustness": hard,
        "compliance_audit": compliance,
        "main_domain": ~compliance,
        "big_team_top20%": Xb["n_agents"].values >= np.quantile(Xb["n_agents"].values, .8),
        "topo_star": Xb["topo_star"].values > 0,
        "topo_pipeline": Xb["topo_pipeline"].values > 0,
        "topo_mesh": Xb["topo_mesh"].values > 0,
        "topo_hierarchical": Xb["topo_hierarchical"].values > 0,
        "topo_blackboard": Xb["topo_blackboard"].values > 0,
    }

    def sl(pred, m):
        idx = np.where(m)[0]
        if len(idx) < 30:
            return None
        return mf1([ys[i] for i in idx], [LABELS[pred[i]] for i in idx])

    # ---------------- scoring ----------------
    ORDER = ["baseline_flat", "exp03_B", "text_word", "text_char", "text_wc",
             "blend_a0.25", "blend_a0.50", "blend_a0.75", "blend_honest"]
    res = {"alpha_policy": "inner split of the train fold (test_size=0.4, seed=%d); "
                           "the validation fold is never used to pick alpha or the text variant"
                           % INNER_SEED,
           "blend_selection": blend_sel, "n_runs": int(n),
           "success_f1_fixed_from_exp03B": SUCCESS_F1, "systems": {}}

    for key in ORDER:
        pred = all_pred[key]
        p = [LABELS[i] for i in pred]
        mf = mf1(ys, p)
        rf = sl(pred, hard)
        turns = [localize(runs[i], p[i]) for i in range(n)]
        h2 = fault_turn_hit_at_k(list(ys), p, list(ft), turns)
        comp = composite({"macro_f1": mf, "robustness_f1": rf,
                          "success_f1": SUCCESS_F1, "fault_turn_hit2": h2})
        d = {"macro_f1": mf, "robustness_f1": rf, "composite": comp,
             "fault_turn_hit2": h2, "success_f1": SUCCESS_F1,
             "per_class": f1_per_class(list(ys), p, LABELS),
             "fold_macro_f1": [float(x) for x in fold_m.get(key, [])],
             "slices": {k: sl(pred, m) for k, m in MASKS.items()}}
        res["systems"][key] = d
        log("  scored %-14s in %.1fs" % (key, time.time() - t0))

    B = res["systems"]["exp03_B"]
    for key in ORDER:
        d = res["systems"][key]
        d["d_macro"] = d["macro_f1"] - B["macro_f1"]
        d["d_rob"] = d["robustness_f1"] - B["robustness_f1"]
        d["d_comp"] = d["composite"] - B["composite"]

    # ---------------- reporting ----------------
    log("")
    log("=" * 92)
    log("HEADLINE  (3-fold OOF; Success F1 fixed from Exp03 B; fault_turn = official localiser)")
    log("=" * 92)
    log("%-15s %8s %9s %9s %8s %9s %9s %9s %9s"
        % ("system", "MacroF1", "RobustF1", "composite", "dMacro", "dRobust",
           "dComp", "no_intent", "longest20"))
    for key in ORDER:
        d = res["systems"][key]
        log("%-15s %8.4f %9.4f %9.4f %+8.4f %+9.4f %+9.4f %9.4f %9.4f"
            % (key, d["macro_f1"], d["robustness_f1"], d["composite"],
               d["d_macro"], d["d_rob"], d["d_comp"],
               d["slices"]["no_intent"], d["slices"]["longest_20%"]))

    log("")
    log("FOLD-BY-FOLD Macro F1")
    for key in ["baseline_flat", "exp03_B", "text_word", "text_char", "text_wc", "blend_honest"]:
        f = res["systems"][key]["fold_macro_f1"]
        log("  %-14s %s  std=%.4f" % (key, ", ".join("%.4f" % x for x in f),
                                      float(np.std(f)) if f else 0.0))

    log("")
    log("PER-CLASS F1")
    cols = ["exp03_B", "text_word", "text_char", "text_wc", "blend_honest"]
    log("%-18s %s" % ("class", "".join("%13s" % k for k in cols)))
    for c in LABELS:
        log("%-18s %s" % (c, "".join("%13.4f" % res["systems"][k]["per_class"][c] for k in cols)))

    log("")
    log("SLICE TABLE  (Macro F1 per slice)")
    log("%-19s %6s %s" % ("slice", "n", "".join("%13s" % k for k in
                                                 ["flat", "exp03B", "word", "char", "wc", "blend"])))
    sk = ["baseline_flat", "exp03_B", "text_word", "text_char", "text_wc", "blend_honest"]
    for name, m in MASKS.items():
        if m.sum() < 30:
            continue
        log("%-19s %6d %s" % (name, int(m.sum()), "".join(
            "%13.4f" % (res["systems"][k]["slices"][name] or 0.0) for k in sk)))

    log("")
    log("LENGTH DECILES")
    dec = np.quantile(nmsg, np.linspace(0, 1, 11))
    log("%-7s %6s %10s %10s %10s %10s %10s" % ("dec", "n", "range", "flat", "exp03B", "word", "blend"))
    dec_rows = []
    for i in range(10):
        lo, hi = dec[i], dec[i + 1]
        m = ((nmsg >= lo) & (nmsg <= hi)) if i == 9 else ((nmsg >= lo) & (nmsg < hi))
        if m.sum() < 30:
            continue
        vals = {k: sl(all_pred[k], m) for k in sk}
        dec_rows.append({"decile": i + 1, "n": int(m.sum()), "lo": float(lo), "hi": float(hi),
                         **{k: v for k, v in vals.items()}})
        log("%-7d %6d %10s %10.4f %10.4f %10.4f %10.4f"
            % (i + 1, int(m.sum()), "%d-%d" % (lo, hi), vals["baseline_flat"],
               vals["exp03_B"], vals["text_word"], vals["blend_honest"]))
    res["deciles"] = dec_rows

    # ---------------- save artifacts BEFORE the fragile diagnostics ----------------
    peak = mem.peak_rss_mb()
    res["resources"] = {"peak_rss_mb": peak, "wall_s": time.time() - _T0,
                        "per_text_system": [
                            {"tag": t, **{k: v for k, v in meta[t][0].items() if k != "names"},
                             "fit_s_mean": float(np.mean([m["fit_s"] for m in meta[t]])),
                             "infer_s_mean": float(np.mean([m["infer_s"] for m in meta[t]]))}
                            for t in TEXT_TAGS]}
    res["coef_note"] = ("per-fold coefficient matrices are ragged (each fold builds its own "
                        "vocabulary), so coefficients cannot be averaged across folds; the "
                        "lexical diagnostic is computed on fold 0 by diagnostics.py")
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o))
    for key in ORDER:
        pd.DataFrame({"run_id": train["run_id"], "y_true": ys,
                      "y_pred": [LABELS[i] for i in all_pred[key]],
                      "success_pred": succ_b, "n_messages": nmsg,
                      "n_intents": Xb["n_intents"].values,
                      "domain": doms}).to_csv(
            os.path.join(HERE, "oof_%s.csv" % key), index=False)
    log("wrote results.json + %d oof files" % len(ORDER))

    # ---------------- lexical diagnostics (fold 0, single vocabulary) ----------------
    log("")
    log("=" * 92)
    log("LEXICAL DIAGNOSTIC (fold 0: one vocabulary per variant, coefficients not averaged)")
    log("=" * 92)
    from sklearn.model_selection import train_test_split
    lex = {}
    for tag in TEXT_TAGS:
        C = coefs[tag][0]
        nm = feat_names[tag]
        if isinstance(nm, tuple):
            nm = np.concatenate(nm)
        log("")
        log("  --- %s (fold 0, %d features) ---" % (tag, C.shape[1]))
        lex[tag] = {"n_features": int(C.shape[1]), "per_class": {}}
        for ci, cname in enumerate(LABELS):
            d = C[ci] - (C.sum(axis=0) - C[ci]) / (len(LABELS) - 1)
            top = list(map(str, nm[np.argsort(-d)[:10]]))
            bot = list(map(str, nm[np.argsort(d)[:5]]))
            lex[tag]["per_class"][cname] = {"top": top, "bottom": bot}
            log("    %-18s + %s" % (cname, ", ".join(top[:8])))
            log("    %-18s - %s" % ("", ", ".join(bot[:3])))
        a = np.abs(C).sum(axis=0)
        srt = np.sort(a)[::-1]
        conc = {int(k): float(srt[:k].sum() / srt.sum()) for k in (20, 100, 1000, 5000)}
        lex[tag]["concentration"] = conc
        log("    coef mass: " + "  ".join("top%-5d %5.1f%%" % (k, 100 * v)
                                          for k, v in conc.items()))
    res["lexical_fold0"] = lex

    log("")
    log("RESOURCES")
    for tag in TEXT_TAGS:
        m0 = meta[tag][0]
        log("  %-10s vocab=%6d  nnz/run=%5.0f  sparse bytes/run=%6.0f  fit=%5.1fs  infer=%5.2fs"
            % (tag, m0["n_features"], m0["nnz"] / m0["rows"],
               m0["bytes"] / m0["rows"], m0["fit_s"], m0["infer_s"]))
    log("  peak process RSS = %.0f MB (%.2f GB)" % (peak, peak / 1024))
    log("  wall clock       = %.0fs (%.1f min)" % (time.time() - _T0, (time.time() - _T0) / 60))
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=lambda o: float(o))


if __name__ == "__main__":
    main()
