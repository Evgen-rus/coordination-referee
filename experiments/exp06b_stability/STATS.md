## Exp06b - paired significance of the seed-level deltas

Seed-level tests: paired t-test and Wilcoxon over the 5 CV splits. n=5 is
tiny, so read these as *consistency* evidence, not as real p-values.
Pooled bootstrap: 20k resamples of runs, all 5 seeds averaged (correlated
rows, so the CI is optimistic). `median` is robust to the outlier seed.

| metric | system | mean d | std | median d | wins | t-test p | Wilcoxon p | pooled-boot CI95 | p |
|---|---|---|---|---|---|---|---|---|---|
| Macro F1 | exp06_ALL | +0.0044 | 0.0029 | +0.0037 | 5/5 | 0.0265 | 0.0625 | [+0.0018, +0.0071] | 0.0010 |
| Macro F1 | full_minus_age | +0.0050 | 0.0013 | +0.0048 | 5/5 | 0.0009 | 0.0625 | [+0.0023, +0.0076] | 0.0004 |
| Macro F1 | full_minus_age_minus_reassign | +0.0036 | 0.0016 | +0.0035 | 5/5 | 0.0069 | 0.0625 | [+0.0012, +0.0060] | 0.0029 |
| Robustness F1 | exp06_ALL | +0.0003 | 0.0115 | -0.0004 | 2/5 | 0.9570 | 1.0000 | [-0.0078, +0.0082] | 0.9462 |
| Robustness F1 | full_minus_age | +0.0018 | 0.0073 | +0.0009 | 3/5 | 0.6060 | 0.8125 | [-0.0059, +0.0096] | 0.6560 |
| Robustness F1 | full_minus_age_minus_reassign | +0.0036 | 0.0095 | +0.0042 | 3/5 | 0.4432 | 0.6250 | [-0.0036, +0.0107] | 0.3256 |
| composite | exp06_ALL | +0.0028 | 0.0038 | +0.0022 | 4/5 | 0.1777 | 0.3125 | - | - |
| composite | full_minus_age | +0.0034 | 0.0025 | +0.0030 | 5/5 | 0.0392 | 0.0625 | - | - |
| composite | full_minus_age_minus_reassign | +0.0031 | 0.0029 | +0.0033 | 4/5 | 0.0768 | 0.1250 | - | - |
| dh F1 overall | exp06_ALL | +0.0122 | 0.0050 | +0.0103 | 5/5 | 0.0054 | 0.0625 | - | - |
| dh F1 overall | full_minus_age | +0.0121 | 0.0055 | +0.0131 | 5/5 | 0.0080 | 0.0625 | - | - |
| dh F1 overall | full_minus_age_minus_reassign | +0.0138 | 0.0050 | +0.0161 | 5/5 | 0.0035 | 0.0625 | - | - |
| long dh RECALL | exp06_ALL | +0.0171 | 0.0163 | +0.0236 | 4/5 | 0.0793 | 0.1250 | [-0.0041, +0.0389] | 0.1162 |
| long dh RECALL | full_minus_age | +0.0094 | 0.0155 | +0.0147 | 4/5 | 0.2456 | 0.4375 | [-0.0112, +0.0307] | 0.3914 |
| long dh RECALL | full_minus_age_minus_reassign | +0.0124 | 0.0237 | +0.0265 | 3/5 | 0.3078 | 0.2500 | [-0.0077, +0.0330] | 0.2338 |
| deadlock F1 | exp06_ALL | +0.0027 | 0.0056 | +0.0027 | 3/5 | 0.3440 | 0.4375 | - | - |
| deadlock F1 | full_minus_age | +0.0042 | 0.0059 | +0.0039 | 4/5 | 0.1900 | 0.1875 | - | - |
| deadlock F1 | full_minus_age_minus_reassign | +0.0045 | 0.0078 | +0.0081 | 3/5 | 0.2687 | 0.3125 | - | - |
| topo_mesh Macro F1 | exp06_ALL | +0.0041 | 0.0057 | +0.0067 | 4/5 | 0.1787 | 0.1875 | [-0.0030, +0.0112] | 0.2510 |
| topo_mesh Macro F1 | full_minus_age | +0.0012 | 0.0062 | +0.0025 | 3/5 | 0.6809 | 0.8125 | [-0.0056, +0.0081] | 0.7244 |
| topo_mesh Macro F1 | full_minus_age_minus_reassign | +0.0046 | 0.0041 | +0.0065 | 4/5 | 0.0690 | 0.1250 | [-0.0015, +0.0106] | 0.1356 |