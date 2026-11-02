"""Exploration strategies for value-based deep RL.

All schedules are expressed on the same horizon (`anneal_steps`) and, for the
proposed Metropolis-Annealed Exploration (MAE), the *rate* of non-greedy actions is
matched to epsilon-greedy so that comparisons isolate *which* actions are explored,
not *how often*.
"""
import math
import numpy as np


def lin(t, T, a, b):
    f = min(1.0, t / max(1, T))
    return a + f * (b - a)


def geo(t, T, a, b):
    f = min(1.0, t / max(1, T))
    return a * (b / a) ** f


class EpsGreedy:
    name = "eps-greedy"

    def __init__(self, nA, anneal_steps, eps0=1.0, eps1=0.02, rng=None):
        self.nA, self.T, self.e0, self.e1 = nA, anneal_steps, eps0, eps1
        self.rng = rng or np.random.default_rng()
        self.t = 0

    def eps(self):
        return lin(self.t, self.T, self.e0, self.e1)

    def act(self, q):
        self.t += 1
        if self.rng.random() < self.eps():
            return int(self.rng.integers(self.nA)), True
        return int(np.argmax(q)), False

    def on_episode(self, ret):
        pass

    def stats(self):
        return {"eps": self.eps()}


class Boltzmann:
    """Softmax exploration with a geometrically annealed temperature (scale-dependent)."""
    name = "boltzmann"

    def __init__(self, nA, anneal_steps, tau0=1.0, tau1=0.01, rng=None):
        self.nA, self.T, self.t0, self.t1 = nA, anneal_steps, tau0, tau1
        self.rng = rng or np.random.default_rng()
        self.t = 0

    def tau(self):
        return geo(self.t, self.T, self.t0, self.t1)

    sigma = 1.0

    def act(self, q):
        self.t += 1
        # scale-free: Q normalised by a running estimate of the action-gap scale
        self.sigma = 0.99 * self.sigma + 0.01 * max(float(q.max() - q.mean()), 1e-8)
        z = (q - q.max()) / (self.sigma * self.tau())
        p = np.exp(z)
        p /= p.sum()
        a = int(self.rng.choice(self.nA, p=p))
        return a, a != int(np.argmax(q))

    def on_episode(self, ret):
        pass

    def stats(self):
        return {"tau": self.tau()}


class MAE:
    """Metropolis-Annealed Exploration (proposed).

    At every step the greedy action a* is the incumbent "state" of an SA chain and a
    candidate a' is drawn uniformly from A\\{a*}.  The candidate is accepted with the
    Metropolis probability  min(1, exp(-Delta / T)),  Delta = (Q(s,a*) - Q(s,a')) / sigma,
    where sigma is a running estimate of the action-gap scale.

    Temperature control (adaptive SA): instead of a hand-tuned temperature schedule,
    log T is driven by stochastic approximation so that the empirical acceptance rate
    tracks a target chi*(t) that decays from chi0 = (|A|-1)/|A| (pure random walk) to
    chi1 (near-greedy), i.e. the same expected non-greedy rate as epsilon-greedy.

    Reheating: if the moving-average training return has not improved for `patience`
    episodes, the target acceptance rate is raised to chi_reheat and decays back with
    half-life `reheat_halflife` steps (bounded number of reheats).
    """
    name = "mae"

    def __init__(self, nA, anneal_steps, chi1=0.02 * 0.75, adaptive=True, reheat=True,
                 T0=1.0, T1=0.01, eta=0.05, ema=0.01, patience_steps=5000, chi_reheat=0.25,
                 reheat_halflife=2000, max_reheats=3, rng=None):
        self.nA, self.Tan = nA, anneal_steps
        self.chi0 = (nA - 1) / nA
        self.chi1 = chi1
        self.adaptive, self.reheat = adaptive, reheat
        self.T0, self.T1 = T0, T1
        self.logT = math.log(T0)
        self.eta, self.ema = eta, ema
        self.acc_hat = self.chi0
        self.sigma = 1.0
        self.rng = rng or np.random.default_rng()
        self.t = 0
        # reheating state
        self.patience, self.chi_r, self.hl, self.max_reheats = patience_steps, chi_reheat, reheat_halflife, max_reheats
        self.best_t = 0
        self.best = -np.inf
        self.ret_ma = None
        self.since_best = 0
        self.boost_t = None
        self.n_reheats = 0
        self.reheat_steps = []

    def target(self):
        # linear target = exactly the non-greedy rate of linearly annealed eps-greedy
        base = lin(self.t, self.Tan, self.chi0, self.chi1)
        if self.boost_t is not None:
            boost = self.chi_r * 0.5 ** ((self.t - self.boost_t) / self.hl)
            base = max(base, boost)
        return base

    def temperature(self):
        if self.adaptive:
            return math.exp(self.logT)
        return geo(self.t, self.Tan, self.T0, self.T1)

    def act(self, q):
        self.t += 1
        a_star = int(np.argmax(q))
        if self.nA == 1:
            return a_star, False
        a_c = int(self.rng.integers(self.nA - 1))
        a_c = a_c + (a_c >= a_star)
        gap = float(q[a_star] - q[a_c])  # >= 0
        # running action-gap scale (makes temperature scale-free)
        self.sigma = (1 - self.ema) * self.sigma + self.ema * max(abs(gap), 1e-8)
        T = self.temperature()
        p_acc = math.exp(-gap / (self.sigma * T + 1e-12))
        acc = self.rng.random() < p_acc
        if self.adaptive:
            self.acc_hat = (1 - self.ema) * self.acc_hat + self.ema * float(acc)
            # Robbins-Monro update on log-temperature (uses p_acc for lower variance)
            self.logT += self.eta * (self.target() - p_acc)
            self.logT = min(max(self.logT, math.log(1e-4)), math.log(1e4))
        else:
            self.acc_hat = (1 - self.ema) * self.acc_hat + self.ema * float(acc)
        return (a_c, True) if acc else (a_star, False)

    def on_episode(self, ret):
        if not self.reheat:
            return
        self.ret_ma = ret if self.ret_ma is None else 0.9 * self.ret_ma + 0.1 * ret
        if self.ret_ma > self.best + 1e-6:
            self.best = self.ret_ma
            self.best_t = self.t
        # reheat after the base schedule has cooled, if the smoothed return has not
        # improved for `patience` environment steps, at most `max_reheats` times
        if (self.t - self.best_t >= self.patience and self.t > self.Tan
                and self.n_reheats < self.max_reheats):
            self.boost_t = self.t
            self.n_reheats += 1
            self.best_t = self.t
            self.reheat_steps.append(self.t)

    def stats(self):
        return {"T": self.temperature(), "acc": self.acc_hat, "target": self.target()}


def make_explorer(method, nA, anneal_steps, rng, total_steps=None):
    chi1 = 0.02 * (nA - 1) / nA  # matches eps1 = 0.02 non-greedy rate
    total_steps = total_steps or int(anneal_steps / 0.3)
    rh = dict(patience_steps=int(0.05 * total_steps), reheat_halflife=int(0.02 * total_steps))
    if method == "eps":
        return EpsGreedy(nA, anneal_steps, rng=rng)
    if method == "boltz":
        return Boltzmann(nA, anneal_steps, rng=rng)
    if method == "mae":
        return MAE(nA, anneal_steps, chi1=chi1, rng=rng, **rh)
    if method == "mae_noreheat":
        return MAE(nA, anneal_steps, chi1=chi1, reheat=False, rng=rng)
    if method == "mae_fixed":  # no acceptance-rate control: geometric T schedule on normalised gaps
        return MAE(nA, anneal_steps, chi1=chi1, adaptive=False, reheat=False, rng=rng)
    if method == "noisy":
        return None
    raise ValueError(method)
