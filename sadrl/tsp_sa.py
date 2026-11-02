"""Simulated annealing driver with block-wise operator selection policies,
including the proposed deep-RL operator selector (trained with MAE exploration)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsp_core import (N_OPS, sa_block, tour_cost, estimate_T0, seed_numba)
from .explore import MAE, EpsGreedy

torch.set_num_threads(1)

# Action set of the operator-selection MDP: operator *mixtures* used within a block.
# 0 = uniform mixture; 1..4 = mixture biased (70%) towards one operator.
MIXES = np.full((N_OPS + 1, N_OPS), 0.1)
MIXES[0] = 1.0 / N_OPS
for _i in range(N_OPS):
    MIXES[_i + 1, _i] = 0.7
N_ACT = N_OPS + 1
STATE_DIM = 6 + N_ACT + N_OPS  # = 15


class SAState:
    def __init__(self):
        self.last_a = -1
        self.op_acc = np.full(N_OPS, 0.5)
        self.since_best = 0

    def features(self, k, K, logT_frac, acc, impr, cur, best):
        f = np.zeros(STATE_DIM, np.float32)
        f[0] = k / K
        f[1] = logT_frac
        f[2] = acc
        f[3] = min(impr * 10, 1.0)
        f[4] = min((cur - best) / best * 10, 1.0)
        f[5] = min(self.since_best / 20, 1.0)
        if self.last_a >= 0:
            f[6 + self.last_a] = 1
        f[6 + N_ACT:] = self.op_acc
        return f


def run_sa(D, policy, seed=0, K=200, moves_per_city=2000, chi0=0.5, t_ratio=3e-3,
           init_tour=None, learner=None, record=False):
    """policy(s) -> (action_id, op_probs).  learner(s, a, r, s2, done, stats) optional."""
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    seed_numba(seed)
    t = rng.permutation(n).astype(np.int64) if init_tour is None else init_tour.copy()
    cur = tour_cost(t, D)
    best, best_t = cur, t.copy()
    T0 = estimate_T0(t, D, 2000, chi0)
    Tf = T0 * t_ratio
    alpha = (Tf / T0) ** (1 / (K - 1))
    L = max(1, int(moves_per_city * n / K))
    st = SAState()
    s = st.features(0, K, 0.0, chi0, 0.0, cur, best)
    trace, acts, probs_hist = [], [], []
    T = T0
    for k in range(K):
        a, probs = policy(s)
        stats = np.zeros((N_OPS, 3))
        best_prev = best
        cur_prev = cur
        cur, best, n_acc, n_impr, n_nb = sa_block(t, D, cur, best_t, best, T, -1, L, probs, stats)
        acc, impr = n_acc / L, n_impr / L
        prop = np.maximum(stats[:, 0], 1)
        seen = stats[:, 0] > 0
        st.op_acc[seen] = 0.7 * st.op_acc[seen] + 0.3 * (stats[seen, 1] / prop[seen])
        st.last_a = a
        st.since_best = 0 if best < best_prev - 1e-9 else st.since_best + 1
        r = 100.0 * math.log(best_prev / best)
        T *= alpha
        s2 = st.features(k + 1, K, (k + 1) / (K - 1), acc, impr, cur, best)
        if learner is not None:
            learner(s, a, r, s2, k == K - 1, stats / max(cur_prev, 1e-12))
        s = s2
        if record:
            trace.append(best)
            acts.append(a)
            probs_hist.append(probs.copy())
    return best, best_t, (np.array(trace), np.array(acts), np.array(probs_hist))


# ----------------------------------------------------------------- policies
def fixed_op(op):
    p = np.zeros(N_OPS)
    p[op] = 1.0
    return lambda s: (-1, p)


def uniform_mix(s):
    return 0, MIXES[0]


class AdaptivePursuit:
    """Adaptive pursuit operator selection (Thierens, GECCO 2005), per-move sampling,
    per-operator credit = accepted improvement per proposal (normalised by tour cost)."""

    def __init__(self, rng, pmin=0.1, alpha=0.3, beta=0.3):
        self.rng, self.pmin, self.a, self.b = rng, pmin, alpha, beta
        self.P = np.full(N_OPS, 1 / N_OPS)
        self.Q = np.zeros(N_OPS)

    def __call__(self, s):
        return -1, self.P.copy()

    def learn(self, s, a, r, s2, done, stats):
        used = stats[:, 0] > 0
        cred = np.zeros(N_OPS)
        cred[used] = stats[used, 2] / stats[used, 0]
        self.Q[used] += self.a * (cred[used] - self.Q[used])
        pmax = 1 - (N_OPS - 1) * self.pmin
        astar = int(np.argmax(self.Q))
        for i in range(N_OPS):
            tgt = pmax if i == astar else self.pmin
            self.P[i] += self.b * (tgt - self.P[i])
        self.P /= self.P.sum()


class QNetSmall(nn.Module):
    def __init__(self, d=STATE_DIM, nA=N_ACT, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, nA))

    def forward(self, x):
        return self.f(x)


class DQNSelector:
    """Double-DQN operator selector; exploration by MAE (proposed) or eps-greedy."""

    def __init__(self, seed=0, explore="mae", anneal_steps=40_000, lr=1e-3, gamma=0.99,
                 buffer=50_000, batch=64, target=1000):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net, self.tgt = QNetSmall(), QNetSmall()
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.batch, self.target = gamma, batch, target
        chi1 = 0.02 * (N_ACT - 1) / N_ACT
        self.exp = (MAE(N_ACT, anneal_steps, chi1=chi1, rng=self.rng, reheat=False) if explore == "mae"
                    else EpsGreedy(N_ACT, anneal_steps, rng=self.rng))
        self.S = np.zeros((buffer, STATE_DIM), np.float32)
        self.S2 = np.zeros((buffer, STATE_DIM), np.float32)
        self.A = np.zeros(buffer, np.int64)
        self.R = np.zeros(buffer, np.float32)
        self.Dn = np.zeros(buffer, np.float32)
        self.cap, self.n, self.i, self.t = buffer, 0, 0, 0
        self.training = True
        self.losses = []

    def q(self, s):
        with torch.no_grad():
            return self.net(torch.from_numpy(s)[None])[0].numpy()

    def __call__(self, s):
        q = self.q(s)
        a = self.exp.act(q)[0] if self.training else int(np.argmax(q))
        return a, MIXES[a]

    def learn(self, s, a, r, s2, done, stats=None):
        j = self.i
        self.S[j], self.A[j], self.R[j], self.S2[j], self.Dn[j] = s, a, r, s2, float(done)
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        self.t += 1
        if self.n < 1000:
            return
        idx = self.rng.integers(self.n, size=self.batch)
        bs, ba, br = torch.from_numpy(self.S[idx]), torch.from_numpy(self.A[idx]), torch.from_numpy(self.R[idx])
        bs2, bd = torch.from_numpy(self.S2[idx]), torch.from_numpy(self.Dn[idx])
        with torch.no_grad():
            a2 = self.net(bs2).argmax(1, keepdim=True)
            y = br + self.gamma * (1 - bd) * self.tgt(bs2).gather(1, a2).squeeze(1)
        qsa = self.net(bs).gather(1, ba[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        if self.t % 100 == 0:
            self.losses.append(float(loss.detach()))
        if self.t % self.target == 0:
            self.tgt.load_state_dict(self.net.state_dict())
