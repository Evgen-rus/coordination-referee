import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution


def test_success_meta_contract_is_seven_raw_label_probabilities():
    assert solution.N_SUCCESS_FEATS == 185
    assert solution.SUCCESS_META_NAMES == ["lp_" + label for label in solution.LABELS]
    assert solution.N_SUCCESS_FEATS + len(solution.SUCCESS_META_NAMES) == 192


def test_label_oof_folds_exclude_each_training_row_once():
    y = np.repeat(np.arange(len(solution.LABELS)), 15)
    folds = solution.label_oof_folds(y)
    seen = np.zeros(len(y), dtype=int)
    for fit, held in folds:
        assert not np.intersect1d(fit, held).size
        seen[held] += 1
    assert np.all(seen == 1)
