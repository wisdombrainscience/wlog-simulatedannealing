"""Numba kernels: SA for the TSP with a portfolio of neighbourhood operators.

Operators (all with O(1) cost-delta evaluation):
  0  2-opt        reverse tour[i+1..j]
  1  swap         exchange the cities at positions i and j
  2  insertion    relocate one city (or-opt with segment length 1)
  3  or-opt       relocate a segment of 2 or 3 consecutive cities
"""
import numpy as np
from numba import njit

N_OPS = 4
OP_NAMES = ["2-opt", "swap", "insert", "or-opt"]


def read_tsplib(path):
    xy, read = [], False
    for line in open(path):
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            read = True
            continue
        if line.startswith("EOF") or not line:
            if read and line.startswith("EOF"):
                break
            continue
        if read:
            p = line.split()
            xy.append((float(p[1]), float(p[2])))
    return np.array(xy)


def dist_matrix(xy, tsplib_round=False):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    if tsplib_round:
        d = np.floor(d + 0.5)
    return d


@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)


@njit(cache=True)
def tour_cost(t, D):
    n = t.shape[0]
    c = 0.0
    for i in range(n):
        c += D[t[i], t[(i + 1) % n]]
    return c


@njit(cache=True)
def _propose(t, D, op):
    """Return (delta, i, j, k, L) for a random move of operator `op` (no state change)."""
    n = t.shape[0]
    if op == 0:  # 2-opt
        while True:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i > j:
                i, j = j, i
            if j - i >= 2 and not (i == 0 and j == n - 1):
                break
        a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
        return D[a, c] + D[b, d] - D[a, b] - D[c, d], i, j, 0, 0
    if op == 1:  # swap
        while True:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i > j:
                i, j = j, i
            if j > i and not (i == 0 and j == n - 1):
                break
        x, y = t[i], t[j]
        pi, nj = t[(i - 1) % n], t[(j + 1) % n]
        if j == i + 1:
            delta = D[pi, y] + D[x, nj] - D[pi, x] - D[y, nj]
        else:
            ni, pj = t[i + 1], t[j - 1]
            delta = (D[pi, y] + D[y, ni] + D[pj, x] + D[x, nj]
                     - D[pi, x] - D[x, ni] - D[pj, y] - D[y, nj])
        return delta, i, j, 0, 0
    # insertion (L=1) or or-opt (L in {2,3})
    if op == 2:
        L = 1
    else:
        L = 2 + np.random.randint(0, 2)
    while True:
        i = np.random.randint(0, n - L + 1)
        k = np.random.randint(0, n)
        # k must not be in {i-1, ..., i+L-1} (mod n)
        rel = (k - (i - 1)) % n
        if rel > L:
            break
    s0, sL = t[i], t[i + L - 1]
    p, q = t[(i - 1) % n], t[(i + L) % n]
    u, v = t[k], t[(k + 1) % n]
    delta = D[p, q] - D[p, s0] - D[sL, q] + D[u, s0] + D[sL, v] - D[u, v]
    return delta, i, k, L, 0


@njit(cache=True)
def _apply(t, op, i, j, L, buf):
    n = t.shape[0]
    if op == 0:
        a, b = i + 1, j
        while a < b:
            t[a], t[b] = t[b], t[a]
            a += 1
            b -= 1
        return
    if op == 1:
        t[i], t[j] = t[j], t[i]
        return


@njit(cache=True)
def _apply_seg(t, i, k, L, buf, seg):
    n = t.shape[0]
    u = t[k]
    for r in range(L):
        seg[r] = t[i + r]
    m = 0
    for r in range(n):
        if r < i or r >= i + L:
            buf[m] = t[r]
            m += 1
    pos = 0
    for r in range(m):
        if buf[r] == u:
            pos = r
            break
    w = 0
    for r in range(pos + 1):
        t[w] = buf[r]
        w += 1
    for r in range(L):
        t[w] = seg[r]
        w += 1
    for r in range(pos + 1, m):
        t[w] = buf[r]
        w += 1


@njit(cache=True)
def sa_block(t, D, cur, best_t, best, T, op, n_moves, op_probs, stats):
    """Run `n_moves` Metropolis moves at fixed temperature T.

    op >= 0 : use that operator;  op == -1 : sample operator per move from op_probs.
    Returns (cur, best, n_acc, n_impr, n_newbest, per-op accepted counts).
    """
    n = t.shape[0]
    buf = np.empty(n, dtype=t.dtype)
    seg = np.empty(3, dtype=t.dtype)
    n_acc = 0
    n_impr = 0
    n_newbest = 0
    for _ in range(n_moves):
        o = op
        if o < 0:
            r = np.random.random()
            acc_p = 0.0
            o = N_OPS_ - 1
            for z in range(N_OPS_):
                acc_p += op_probs[z]
                if r < acc_p:
                    o = z
                    break
        delta, i, j, L, _x = _propose(t, D, o)
        stats[o, 0] += 1
        if delta < 0.0 or np.random.random() < np.exp(-delta / T):
            stats[o, 1] += 1
            if delta < 0.0:
                stats[o, 2] -= delta
            if o <= 1:
                _apply(t, o, i, j, L, buf)
            else:
                _apply_seg(t, i, j, L, buf, seg)
            cur += delta
            n_acc += 1
            if delta < -1e-9:
                n_impr += 1
            if cur < best - 1e-9:
                best = cur
                best_t[:] = t
                n_newbest += 1
    return cur, best, n_acc, n_impr, n_newbest


N_OPS_ = 4


@njit(cache=True)
def nearest_neighbour(D, start):
    n = D.shape[0]
    vis = np.zeros(n, np.bool_)
    t = np.empty(n, np.int64)
    t[0] = start
    vis[start] = True
    for k in range(1, n):
        a = t[k - 1]
        bj, bd = -1, 1e300
        for j in range(n):
            if not vis[j] and D[a, j] < bd:
                bd, bj = D[a, j], j
        t[k] = bj
        vis[bj] = True
    return t


@njit(cache=True)
def two_opt_ls(t, D):
    """First-improvement 2-opt local search until a local optimum."""
    n = t.shape[0]
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-10:
                    x, y = i + 1, j
                    while x < y:
                        t[x], t[y] = t[y], t[x]
                        x += 1
                        y -= 1
                    improved = True
    return t


@njit(cache=True)
def estimate_T0(t, D, n_samples, chi0):
    """Initial temperature so that mean uphill 2-opt move is accepted w.p. chi0."""
    s = 0.0
    c = 0
    for _ in range(n_samples):
        delta, i, j, L, _x = _propose(t, D, 0)
        if delta > 0:
            s += delta
            c += 1
    m = s / max(c, 1)
    return -m / np.log(chi0)
