import json
import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_common import *  # noqa
from sadrl.tsp_core import read_tsplib, dist_matrix  # noqa

R = json.load(open("results/tsp/eval_tsplib.json"))
ORT = json.load(open("results/tsp/ortools_10s.json"))
INST = list(R)
SA = ["SA-2opt", "SA-insert", "SA-mix", "SA-AP", "SA-DQN-eps", "SA-DQN-MAE"]
SA_ALL = ["SA-2opt", "SA-swap", "SA-insert", "SA-oropt", "SA-mix", "SA-AP", "SA-DQN-eps", "SA-DQN-MAE"]
LBL = {"SA-2opt": "SA-2opt", "SA-swap": "SA-swap", "SA-insert": "SA-Ins.", "SA-oropt": "SA-Or",
       "SA-mix": "SA-Unif.", "SA-AP": "SA-AP", "SA-DQN-eps": "DQN($\\epsilon$)", "SA-DQN-MAE": "DQN(MAE)"}
COLS = {"SA-2opt": "#eb6834", "SA-mix": "#1baf7a", "SA-AP": "#eda100", "SA-DQN-eps": "#e87ba4",
        "SA-DQN-MAE": "#2a78d6", "SA-insert": "#6250d6"}
os.makedirs("results/tables", exist_ok=True)


def gap(v, o):
    return 100 * (np.asarray(v) / o - 1)


out = {}
for B in [2000, 8000]:
    G = np.array([[np.mean(gap(R[i][f"{m}@{B}"], R[i]["opt"])) for m in SA_ALL] for i in INST])
    out[B] = G
    ranks = np.array([rankdata(g) for g in G[:, [SA_ALL.index(m) for m in SA]]])
    fr = friedmanchisquare(*[G[:, SA_ALL.index(m)] for m in SA])
    print(f"B={B}: mean gap", dict(zip(SA_ALL, G.mean(0).round(2))))
    print("  avg rank", dict(zip(SA, ranks.mean(0).round(2))), "Friedman p=%.2g" % fr.pvalue)
    # Holm-corrected Wilcoxon: DQN-MAE vs others
    ps = []
    for m in SA:
        if m == "SA-DQN-MAE":
            continue
        p = wilcoxon(G[:, SA_ALL.index("SA-DQN-MAE")], G[:, SA_ALL.index(m)]).pvalue
        ps.append((m, p))
    ps.sort(key=lambda x: x[1])
    k = len(ps)
    holm = []
    run = 0
    for j, (m, p) in enumerate(ps):
        run = max(run, min(1, (k - j) * p))
        holm.append((m, p, run))
    print("  Wilcoxon vs DQN-MAE (raw, Holm):", [(m, round(p, 4), round(h, 4)) for m, p, h in holm])
    out[f"stats{B}"] = {"rank": dict(zip(SA, ranks.mean(0).tolist())), "friedman_p": float(fr.pvalue),
                        "holm": holm, "mean": dict(zip(SA_ALL, G.mean(0).tolist()))}

# ---- LaTeX table: per-instance gaps at both budgets for key methods
keys = ["SA-2opt", "SA-mix", "SA-AP", "SA-DQN-eps", "SA-DQN-MAE"]
lines = []
for i in INST:
    r = R[i]
    o = r["opt"]
    row = [i, str(r["n"]), f"{gap(r['NN+2opt'], o)[0]:.2f}", f"{gap([ORT[i]], o)[0]:.2f}"]
    for B in [2000, 8000]:
        vals = {m: np.mean(gap(r[f"{m}@{B}"], o)) for m in keys}
        bm = min(vals, key=vals.get)
        for m in keys:
            s = f"{vals[m]:.2f}"
            row.append(f"\\textbf{{{s}}}" if m == bm else s)
    lines.append(" & ".join(row) + r" \\")
avg = ["\\midrule Mean", "", f"{np.mean([gap(R[i]['NN+2opt'], R[i]['opt'])[0] for i in INST]):.2f}",
       f"{np.mean([gap([ORT[i]], R[i]['opt'])[0] for i in INST]):.2f}"]
rk = ["Avg. rank", "", "", ""]
for B in [2000, 8000]:
    G = out[B][:, [SA_ALL.index(m) for m in keys]]
    mm = G.mean(0)
    rr = np.array([rankdata(g) for g in G]).mean(0)
    for j in range(len(keys)):
        avg.append(f"\\textbf{{{mm[j]:.2f}}}" if j == int(np.argmin(mm)) else f"{mm[j]:.2f}")
        rk.append(f"\\textbf{{{rr[j]:.2f}}}" if j == int(np.argmin(rr)) else f"{rr[j]:.2f}")
lines.append(" & ".join(avg) + r" \\")
lines.append(" & ".join(rk) + r" \\")
open("results/tables/tsp_main.tex", "w").write("\n".join(lines) + "\n")

# single-operator table (mean gap over instances)
so = []
for B in [2000, 8000]:
    G = out[B]
    so.append(" & ".join([f"$B={B}$"] + [f"{G[:, SA_ALL.index(m)].mean():.2f}" for m in SA_ALL]) + r" \\")
open("results/tables/tsp_ops.tex", "w").write("\n".join(so) + "\n")
json.dump({str(k): v for k, v in out.items() if str(k).startswith("stats")}, open("results/tables/tsp_stats.json", "w"),
          indent=1, default=float)

# ---- convergence figure (B=8000 traces)
show = ["kroA100", "ch150", "kroA200", "a280"]
fig, axs = plt.subplots(1, 4, figsize=(COL2, 1.65))
for ax, i in zip(axs, show):
    o = R[i]["opt"]
    for m in ["SA-2opt", "SA-mix", "SA-AP", "SA-DQN-eps", "SA-DQN-MAE"]:
        tr = gap(R[i][f"{m}@8000_trace"], o)
        x = np.arange(1, len(tr) + 1)
        ax.plot(x, tr, color=COLS[m], lw=1.5 if m == "SA-DQN-MAE" else 1.0, label=LBL[m],
                ls="-" if m == "SA-DQN-MAE" else "--" if m == "SA-2opt" else "-." if m == "SA-mix" else ":" if m == "SA-AP" else (0, (5, 1, 1, 1)))
    ax.set_yscale("log")
    ax.set_xlim(60, 200)
    ax.set_title(f"{i} ($n={R[i]['n']}$)")
    ax.set_xlabel("temperature block $k$")
axs[0].set_ylabel("gap to optimum (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1))
fig.tight_layout(w_pad=0.5)
fig.savefig("figs/tsp_convergence.pdf")
plt.close(fig)

# ---- learned policy: expected operator probabilities vs annealing progress (all instances, B=8000)
P = []
for i in INST:
    for run in R[i]["SA-DQN-MAE@8000_ops"]:
        P.append(np.array(run))  # (20, 4) subsampled every 10 blocks
P = np.array(P)  # runs x 20 x 4
mp = P.mean(0)
fig, ax = plt.subplots(figsize=(COL1, 1.6))
x = np.arange(mp.shape[0]) * 10 + 5
names = ["2-opt", "swap", "insertion", "or-opt"]
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ax.stackplot(x, mp.T, labels=names, colors=cols, alpha=0.9, edgecolor="#fcfcfb", linewidth=0.6)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, 1)
ax.set_xlabel("temperature block $k$ (hot $\\rightarrow$ cold)")
ax.set_ylabel("operator prob.")
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.28), fontsize=6.5, handlelength=1)
ax.grid(False)
fig.savefig("figs/tsp_policy.pdf")
plt.close(fig)

# ---- training curves of selectors
fig, ax = plt.subplots(figsize=(COL1, 1.55))
for e, col, lab in [("mae", "#2a78d6", "MAE"), ("eps", "#eb6834", "$\\epsilon$-greedy")]:
    Y = []
    for s in range(3):
        f = f"results/tsp/train_{e}_s{s}.json"
        if os.path.exists(f):
            h = json.load(open(f))["hist"]
            Y.append([x["ratio_vs_nn2opt"] for x in h])
    Y = np.array(Y)
    w = 25
    sm = np.array([np.convolve(y, np.ones(w) / w, mode="valid") for y in Y])
    xx = np.arange(sm.shape[1]) + w
    ax.plot(xx, sm.mean(0), color=col, label=f"selector trained with {lab}")
    ax.fill_between(xx, sm.min(0), sm.max(0), color=col, alpha=0.15, lw=0)
ax.set_xlabel("training episode (random instance)")
ax.set_ylabel("cost / cost(NN+2opt)")
ax.legend(fontsize=6.5)
fig.savefig("figs/tsp_training.pdf")
plt.close(fig)

# ---- best tour plot (kroA200, DQN-MAE vs SA-2opt) -- recomputed for one seed
from sadrl.tsp_sa import run_sa, fixed_op  # noqa
from tsp_exp import load_sel  # noqa
xy = read_tsplib("data/tsplib/kroA200.tsp")
D = dist_matrix(xy, True)
sel = load_sel("results/tsp/sel_mae_s0.pt")
fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.75))
for ax, (pol, name) in zip(axs, [(fixed_op(0), "SA-2opt"), (sel, "SA-DQN(MAE)")]):
    b, t, _ = run_sa(D, pol, seed=0, moves_per_city=8000)
    tt = np.r_[t, t[0]]
    ax.plot(xy[tt, 0], xy[tt, 1], color="#2a78d6" if "DQN" in name else "#eb6834", lw=0.7)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color="#0b0b0b", zorder=3)
    ax.set_title(f"{name}: {b:.0f} (gap {100 * (b / 29368 - 1):.2f}%)", fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout(w_pad=0.2)
fig.savefig("figs/tsp_tours.pdf")
plt.close(fig)
print("figures done")
