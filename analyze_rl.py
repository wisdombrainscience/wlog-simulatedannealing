"""Aggregate RL results -> tables (LaTeX) + figures."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_common import *  # noqa

ENVS = ["DynamicMaze", "CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
ENV_LABEL = {"DynamicMaze": "DynamicMaze", "CartPole-v1": "CartPole-v1", "Acrobot-v1": "Acrobot-v1",
             "LunarLander-v3": "LunarLander-v3"}
MAIN = ["mae", "eps", "boltz", "noisy"]
ABL = ["mae", "mae_noreheat", "mae_fixed", "eps"]
OUT = "results/tables"
os.makedirs(OUT, exist_ok=True)

R = defaultdict(dict)
for f in glob.glob("results/rl/*.json"):
    env, meth, sd = os.path.basename(f)[:-5].split("__")
    R[(env, meth)][int(sd[1:])] = json.load(open(f))


def final(L):
    return float(np.mean(L["eval_ret"][-3:]))


def auc(L):
    return float(np.mean(L["eval_ret"]))


def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x)
    if len(x) < 2:
        return x.mean(0), x.mean(0)
    idx = rng.integers(len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5, 0), np.percentile(m, 97.5, 0)


summary = {}
for env in ENVS:
    for m in set(MAIN + ABL):
        runs = R.get((env, m), {})
        if not runs:
            continue
        summary[(env, m)] = {"final": [final(L) for L in runs.values()], "auc": [auc(L) for L in runs.values()],
                             "seeds": sorted(runs)}


def fmt(v, best=False, d=1):
    s = f"{np.mean(v):.{d}f}$\\pm${np.std(v):.{d}f}"
    return f"\\textbf{{{s}}}" if best else s


def pstar(a, b):
    if len(a) < 3 or len(b) < 3:
        return ""
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    return "$^{\\dagger}$" if p < 0.05 else ""


def table(methods, name, caption_metric):
    lines = []
    for env in ENVS:
        d = 2 if env == "DynamicMaze" else 1
        row = [ENV_LABEL[env]]
        for metric in ["final", "auc"]:
            vals = {m: summary.get((env, m), {}).get(metric) for m in methods}
            avail = {m: v for m, v in vals.items() if v}
            if not avail:
                row += ["--"] * len(methods)
                continue
            bm = max(avail, key=lambda m: np.mean(avail[m]))
            for m in methods:
                v = vals[m]
                if not v:
                    row.append("--")
                    continue
                mark = pstar(vals["mae"], v) if m != "mae" and vals.get("mae") else ""
                row.append(fmt(v, m == bm, d) + mark)
        lines.append(" & ".join(row) + r" \\")
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines) + "\n")
    return lines


def prob_improvement(m1, m2, metric="auc"):
    """P(X_m1 > X_m2) averaged over envs (Agarwal et al., 2021), ties count 1/2."""
    ps = []
    for env in ENVS:
        a = summary.get((env, m1), {}).get(metric)
        b = summary.get((env, m2), {}).get(metric)
        if not a or not b:
            continue
        a, b = np.array(a)[:, None], np.array(b)[None, :]
        ps.append(((a > b) + 0.5 * (a == b)).mean())
    return float(np.mean(ps)) if ps else float("nan"), len(ps)


def curves(methods, fname, envs=ENVS, ncol=4, height=1.75):
    fig, axs = plt.subplots(1, len(envs), figsize=(COL2, height))
    for ax, env in zip(np.atleast_1d(axs), envs):
        for m in methods:
            runs = R.get((env, m), {})
            if not runs:
                continue
            ts = np.array(next(iter(runs.values()))["eval_t"]) / 1000
            Y = np.array([L["eval_ret"] for L in runs.values()])
            lo, hi = boot_ci(Y)
            ax.fill_between(ts, lo, hi, color=C[m], alpha=0.15, lw=0)
            ax.plot(ts, Y.mean(0), color=C[m], ls=LS[m], label=LABEL[m], lw=1.6 if m == "mae" else 1.1)
        ax.set_title(ENV_LABEL[env])
        ax.set_xlabel("environment steps ($\\times 10^3$)")
    np.atleast_1d(axs)[0].set_ylabel("greedy evaluation return")
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f"figs/{fname}.pdf")
    plt.close(fig)


def maze_success(fname):
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.55))
    bins = np.linspace(0, 100_000, 21)
    for m in MAIN:
        runs = R.get(("DynamicMaze", m), {})
        if not runs:
            continue
        S, Cc = [], []
        for L in runs.values():
            t, r = np.array(L["ep_t"]), np.array(L["ep_ret"])
            b = np.digitize(t, bins) - 1
            S.append([np.mean(r[b == i] > 0) if np.any(b == i) else np.nan for i in range(20)])
            Cc.append([np.mean((r[b == i] <= -1) & (r[b == i] > -1.49)) if np.any(b == i) else np.nan
                       for i in range(20)])
        x = (bins[:-1] + bins[1:]) / 2000
        axs[0].plot(x, np.nanmean(S, 0), color=C[m], ls=LS[m], label=LABEL[m], lw=1.5 if m == "mae" else 1)
        axs[1].plot(x, np.nanmean(Cc, 0), color=C[m], ls=LS[m], lw=1.5 if m == "mae" else 1)
    axs[0].set_title("(a) success rate", loc="left")
    axs[1].set_title("(b) collision rate", loc="left")
    for ax in axs:
        ax.set_xlabel("steps ($\\times 10^3$)")
        ax.set_ylim(-0.02, 1.02)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=6, handlelength=1.6,
               columnspacing=0.8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f"figs/{fname}.pdf")
    plt.close(fig)


def mae_dynamics(fname):
    fig, axs = plt.subplots(2, 2, figsize=(COL1, 2.6), sharex="col")
    for j, env in enumerate(["DynamicMaze", "CartPole-v1"]):
        runs = R.get((env, "mae"), {})
        eps_runs = R.get((env, "eps"), {})
        if not runs:
            continue
        L = runs[sorted(runs)[0]]
        t = np.array(L["st_t"]) / 1000
        tgt = np.array([d["target"] for d in L["st"]])
        T = np.array([d["T"] for d in L["st"]])
        ng = np.array(L["nongreedy"])
        axs[0, j].plot(t, tgt, color=C["ink2"], lw=0.9, ls="--", label="target $\\chi^*(t)$")
        axs[0, j].plot(t, ng, color=C["mae"], lw=0.9, label="realised (MAE)")
        if eps_runs:
            Le = eps_runs[sorted(eps_runs)[0]]
            nA = 4 if env == "DynamicMaze" else 2  # random draws -> expected non-greedy actions
            axs[0, j].plot(np.array(Le["st_t"]) / 1000, np.array(Le["nongreedy"]) * (nA - 1) / nA, color=C["eps"],
                           lw=0.7, alpha=0.8,
                           label="realised ($\\epsilon$-greedy)")
        axs[1, j].semilogy(t, T, color=C["mae"], lw=1)
        for rh in L.get("reheats", []):
            for ax in axs[:, j]:
                ax.axvline(rh / 1000, color="#e34948", lw=0.7, ls=":")
        axs[0, j].set_title(ENV_LABEL[env])
        axs[1, j].set_xlabel("steps ($\\times 10^3$)")
    axs[0, 0].set_ylabel("non-greedy rate")
    axs[1, 0].set_ylabel("temperature $T_t$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), fontsize=6, handlelength=1.5,
               columnspacing=0.8)
    fig.tight_layout(h_pad=0.3, w_pad=0.6)
    fig.savefig(f"figs/{fname}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    print("runs:", {f"{e}|{m}": len(v) for (e, m), v in sorted(R.items())})
    for ln in table(MAIN, "rl_main", ""):
        print(ln)
    print()
    for ln in table(ABL, "rl_ablation", ""):
        print(ln)
    pi = {m: prob_improvement("mae", m) for m in ["eps", "boltz", "noisy", "mae_noreheat", "mae_fixed"]}
    pi_final = {m: prob_improvement("mae", m, "final") for m in ["eps", "boltz", "noisy", "mae_noreheat", "mae_fixed"]}
    print("P(MAE > X) auc:", pi)
    print("P(MAE > X) final:", pi_final)
    # p-values detail
    pv = {}
    for env in ENVS:
        for m in ["eps", "boltz", "noisy", "mae_noreheat", "mae_fixed"]:
            a = summary.get((env, "mae"), {}).get("auc")
            b = summary.get((env, m), {}).get("auc")
            if a and b and len(a) > 2 and len(b) > 2:
                pv[f"{env}|{m}"] = (round(float(np.mean(a)), 3), round(float(np.mean(b)), 3),
                                    float(mannwhitneyu(a, b).pvalue))
    for k, v in pv.items():
        print(k, v)
    json.dump({"pi_auc": pi, "pi_final": pi_final, "pvals": pv,
               "summary": {f"{e}|{m}": v for (e, m), v in summary.items()},
               "wall": {f"{e}|{m}": float(np.mean([L["wall"] for L in R[(e, m)].values()])) for (e, m) in R}},
              open(f"{OUT}/rl_summary.json", "w"), indent=1)
    curves(MAIN, "rl_curves")
    curves(ABL, "rl_ablation_curves", envs=ENVS[:3])
    maze_success("maze_success")
    mae_dynamics("mae_dynamics")
