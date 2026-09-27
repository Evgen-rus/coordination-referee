"""Unit test for the iterative Tarjan SCC in ``lifecycle.py``.

Compares against a straightforward recursive reference on many random graphs,
including self-loops, disconnected parts and complete digraphs.
"""

from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lifecycle import _sccs  # noqa: E402


def ref_scc(nodes, edges):
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
    index, low, on, out = {}, {}, {}, []
    counter = [0]
    stack: list = []

    def go(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        on[v] = True
        stack.append(v)
        for w in adj[v]:
            if w not in index:
                go(w)
                low[v] = min(low[v], low[w])
            elif on.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on[w] = False
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for r in nodes:
        if r not in index:
            go(r)
    return sorted(tuple(sorted(c)) for c in out if c)


def norm(comps):
    return sorted(tuple(sorted(c)) for c in comps if c)


def main() -> int:
    random.seed(0)
    bad = 0
    trials = 0
    for _ in range(2000):
        k = random.randint(1, 9)
        nodes = [chr(97 + i) for i in range(k)]
        edges = [(random.choice(nodes), random.choice(nodes))
                 for _ in range(random.randint(0, k * 3))]
        trials += 1
        got, exp = norm(_sccs(nodes, edges)), ref_scc(nodes, edges)
        if got != exp:
            bad += 1
            if bad <= 3:
                print("MISMATCH nodes=%s edges=%s\n  got %s\n  exp %s"
                      % (nodes, edges, got, exp))
    # explicit stress cases
    stress = [
        ([], []),
        (["a"], []),
        (["a"], [("a", "a")]),
        (["a", "b"], [("a", "b"), ("b", "a")]),
        ([chr(97 + i) for i in range(7)],
         [(chr(97 + i), chr(97 + j)) for i in range(7) for j in range(7)]),
        (list("abcdef"), [("a", "b"), ("c", "d"), ("e", "f")]),
        (list("abcde"), [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")]),
    ]
    for nodes, edges in stress:
        trials += 1
        got, exp = norm(_sccs(nodes, edges)), ref_scc(nodes, edges)
        if got != exp:
            bad += 1
            print("STRESS MISMATCH %s %s: got %s exp %s" % (nodes, edges, got, exp))
    print("SCC unit test: %d trials, %d mismatches" % (trials, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
