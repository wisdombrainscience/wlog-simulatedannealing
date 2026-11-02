"""Double DQN with pluggable exploration (incl. NoisyNet baseline)."""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import make_env
from .explore import make_explorer

torch.set_num_threads(1)


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2018)."""

    def __init__(self, i, o, sigma0=0.5):
        super().__init__()
        self.i, self.o = i, o
        self.mu_w = nn.Parameter(torch.empty(o, i))
        self.sig_w = nn.Parameter(torch.empty(o, i))
        self.mu_b = nn.Parameter(torch.empty(o))
        self.sig_b = nn.Parameter(torch.empty(o))
        r = 1 / math.sqrt(i)
        nn.init.uniform_(self.mu_w, -r, r)
        nn.init.uniform_(self.mu_b, -r, r)
        nn.init.constant_(self.sig_w, sigma0 / math.sqrt(i))
        nn.init.constant_(self.sig_b, sigma0 / math.sqrt(i))
        self.register_buffer("eps_i", torch.zeros(i))
        self.register_buffer("eps_o", torch.zeros(o))
        self.noise = True
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_i.normal_()
        self.eps_o.normal_()

    def forward(self, x):
        if self.noise:
            ei, eo = self._f(self.eps_i), self._f(self.eps_o)
            w = self.mu_w + self.sig_w * torch.outer(eo, ei)
            b = self.mu_b + self.sig_b * eo
        else:
            w, b = self.mu_w, self.mu_b
        return F.linear(x, w, b)


class QNet(nn.Module):
    def __init__(self, d, nA, h=128, noisy=False):
        super().__init__()
        self.noisy = noisy
        L = NoisyLinear if noisy else nn.Linear
        self.l1 = nn.Linear(d, h)
        self.l2 = L(h, h)
        self.l3 = L(h, nA)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def set_noise(self, on):
        if self.noisy:
            self.l2.noise = on
            self.l3.noise = on

    def reset_noise(self):
        if self.noisy:
            self.l2.reset_noise()
            self.l3.reset_noise()


class Replay:
    def __init__(self, cap, d, rng):
        self.s = np.zeros((cap, d), np.float32)
        self.s2 = np.zeros((cap, d), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.n, self.i, self.rng = cap, 0, 0, rng

    def add(self, s, a, r, s2, d):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = self.rng.integers(self.n, size=b)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.d[idx]))


HP = {
    "DynamicMaze": dict(steps=100_000, lr=5e-4, gamma=0.99, buffer=50_000, batch=64, start=1000,
                        target=1000, anneal=0.4, eval_every=5000, eval_eps=10, h=128),
    "CartPole-v1": dict(steps=100_000, lr=5e-4, gamma=0.99, buffer=50_000, batch=64, start=1000,
                        target=500, anneal=0.3, eval_every=5000, eval_eps=5, h=128),
    "Acrobot-v1": dict(steps=100_000, lr=5e-4, gamma=0.99, buffer=50_000, batch=64, start=1000,
                       target=500, anneal=0.3, eval_every=5000, eval_eps=5, h=128),
    "MountainCar-v0": dict(steps=150_000, lr=1e-3, gamma=0.99, buffer=100_000, batch=64, start=1000,
                           target=500, anneal=0.3, eval_every=7500, eval_eps=5, h=128),
    "LunarLander-v3": dict(steps=200_000, lr=5e-4, gamma=0.99, buffer=100_000, batch=64, start=5000,
                           target=1000, anneal=0.3, eval_every=10000, eval_eps=5, h=128),
}


def evaluate(net, env, n):
    net.set_noise(False)
    rets = []
    for _ in range(n):
        s, done, R = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                a = int(net(torch.from_numpy(s)[None]).argmax())
            s, r, term, trunc = env.step(a)
            R += r
            done = term or trunc
        rets.append(R)
    net.set_noise(True)
    return float(np.mean(rets))


def run(env_name, method, seed, steps=None, log_every=1000, overrides=None):
    hp = dict(HP[env_name])
    if overrides:
        hp.update(overrides)
    if steps:
        hp["steps"] = steps
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = make_env(env_name, seed)
    env.seed(seed)
    eval_env = make_env(env_name, seed + 10_000)
    eval_env.seed(seed + 10_000)
    d, nA = env.obs_dim, env.n_actions
    noisy = method == "noisy"
    net, tgt = QNet(d, nA, hp["h"], noisy), QNet(d, nA, hp["h"], noisy)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    buf = Replay(hp["buffer"], d, rng)
    explorer = make_explorer(method, nA, int(hp["anneal"] * hp["steps"]), rng, hp["steps"])

    log = {"eval_t": [], "eval_ret": [], "ep_t": [], "ep_ret": [], "st_t": [], "st": [], "nongreedy": []}
    s, R, ng_cnt = env.reset(), 0.0, 0
    t0 = time.time()
    for t in range(1, hp["steps"] + 1):
        with torch.no_grad():
            if noisy:
                net.reset_noise()
            q = net(torch.from_numpy(s)[None])[0].numpy()
        if explorer is None:
            a, ng = int(np.argmax(q)), False
        else:
            a, ng = explorer.act(q)
        ng_cnt += ng
        s2, r, term, trunc = env.step(a)
        buf.add(s, a, r, s2, float(term))
        R += r
        s = s2
        if term or trunc:
            log["ep_t"].append(t)
            log["ep_ret"].append(R)
            if explorer is not None:
                explorer.on_episode(R)
            s, R = env.reset(), 0.0
        if t >= hp["start"] and t % hp.get("train_freq", 4) == 0:
            bs, ba, br, bs2, bd = buf.sample(hp["batch"])
            if noisy:
                net.reset_noise(); tgt.reset_noise()
            with torch.no_grad():
                a2 = net(bs2).argmax(1, keepdim=True)
                y = br + hp["gamma"] * (1 - bd) * tgt(bs2).gather(1, a2).squeeze(1)
            qsa = net(bs).gather(1, ba[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(qsa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
        if t % hp["target"] == 0:
            tgt.load_state_dict(net.state_dict())
        if t % log_every == 0:
            log["st_t"].append(t)
            log["st"].append(explorer.stats() if explorer is not None else {})
            log["nongreedy"].append(ng_cnt / log_every)
            ng_cnt = 0
        if t % hp["eval_every"] == 0:
            log["eval_t"].append(t)
            log["eval_ret"].append(evaluate(net, eval_env, hp["eval_eps"]))
    log["wall"] = time.time() - t0
    log["reheats"] = getattr(explorer, "reheat_steps", [])
    log["hp"] = hp
    return log
