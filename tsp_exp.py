"""TSP experiments: train DQN operator selectors, evaluate on TSPLIB."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sadrl.tsp_core import read_tsplib, dist_matrix, nearest_neighbour, two_opt_ls, tour_cost  # noqa
from sadrl.tsp_sa import run_sa, fixed_op, uniform_mix, AdaptivePursuit, DQNSelector, N_ACT  # noqa

OPT = {"eil51": 426, "berlin52": 7542, "st70": 675, "eil76": 538, "pr76": 108159, "rat99": 1211,
       "kroA100": 21282, "kroB100": 22141, "eil101": 629, "lin105": 14379, "ch130": 6110, "ch150": 6528,
       "kroA150": 26524, "kroA200": 29368, "tsp225": 3916, "a280": 2579}
K = 200


def train(seed, explore, episodes, out, mpc=2000):
    rng = np.random.default_rng(1000 + seed)
    sel = DQNSelector(seed=seed, explore=explore, anneal_steps=int(0.6 * episodes * K))
    hist = []
    t0 = time.time()
    for ep in range(episodes):
        n = int(rng.integers(50, 151))
        D = dist_matrix(rng.random((n, 2)) * 1000)
        best, _, (tr, ops, _p) = run_sa(D, sel, seed=int(rng.integers(1 << 30)), K=K, moves_per_city=mpc,
                                    learner=sel.learn, record=True)
        nn_cost = tour_cost(two_opt_ls(nearest_neighbour(D, 0), D), D)
        hist.append({"ep": ep, "n": n, "ratio_vs_nn2opt": best / nn_cost,
                     "act_freq": np.bincount(ops, minlength=N_ACT).tolist()})
        if ep % 25 == 0:
            print(explore, seed, ep, round(float(np.mean([h["ratio_vs_nn2opt"] for h in hist[-25:]])), 4),
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save(sel.net.state_dict(), f"{out}/sel_{explore}_s{seed}.pt")
    json.dump({"hist": hist, "losses": sel.losses}, open(f"{out}/train_{explore}_s{seed}.json", "w"))


def load_sel(path):
    sel = DQNSelector(seed=0)
    sel.net.load_state_dict(torch.load(path))
    sel.training = False
    return sel


def evaluate(out, runs, mpcs, inst_names, n_models):
    res = {}
    sels = {e: [load_sel(f"{out}/sel_{e}_s{m}.pt") for m in range(n_models)] for e in ["mae", "eps"]}
    for name in inst_names:
        D = dist_matrix(read_tsplib(f"data/tsplib/{name}.tsp"), tsplib_round=True)
        n = D.shape[0]
        r = {"opt": OPT[name], "n": n}
        t0 = time.time()
        nn = nearest_neighbour(D, 0)
        r["NN"] = [tour_cost(nn, D)]
        r["NN+2opt"] = [tour_cost(two_opt_ls(nn.copy(), D), D)]
        for mpc in mpcs:
            for meth in ["SA-2opt", "SA-swap", "SA-insert", "SA-oropt", "SA-mix", "SA-AP", "SA-DQN-eps", "SA-DQN-MAE"]:
                vals, traces, opsl = [], [], []
                for sd in range(runs):
                    learner = None
                    if meth == "SA-2opt":
                        pol = fixed_op(0)
                    elif meth == "SA-swap":
                        pol = fixed_op(1)
                    elif meth == "SA-insert":
                        pol = fixed_op(2)
                    elif meth == "SA-oropt":
                        pol = fixed_op(3)
                    elif meth == "SA-mix":
                        pol = uniform_mix
                    elif meth == "SA-AP":
                        pol = AdaptivePursuit(np.random.default_rng(sd))
                        learner = pol.learn
                    elif meth == "SA-DQN-eps":
                        pol = sels["eps"][sd % n_models]
                    else:
                        pol = sels["mae"][sd % n_models]
                    b, _, (tr, ops, pr) = run_sa(D, pol, seed=sd, K=K, moves_per_city=mpc, learner=learner,
                                                 record=True)
                    vals.append(float(b))
                    traces.append(tr.tolist())
                    opsl.append(pr.mean(0).tolist() if False else pr[::10].tolist())
                r[f"{meth}@{mpc}"] = vals
                if mpc == mpcs[len(mpcs) // 2]:
                    r[f"{meth}@{mpc}_trace"] = np.mean(traces, 0).tolist()
                    r[f"{meth}@{mpc}_ops"] = opsl
        res[name] = r
        g = {k: 100 * (np.mean(v) / OPT[name] - 1) for k, v in r.items()
             if isinstance(v, list) and not k.endswith(("_trace", "_ops"))}
        mid = mpcs[len(mpcs) // 2]
        print(name, f"{time.time() - t0:.0f}s", {k: round(v, 2) for k, v in g.items() if k.endswith(f"@{mid}") or "NN" in k},
              flush=True)
        json.dump(res, open(f"{out}/eval_tsplib.json", "w"))


def ortools_ref(out, inst_names, secs):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    res = {}
    for name in inst_names:
        D = dist_matrix(read_tsplib(f"data/tsplib/{name}.tsp"), tsplib_round=True).astype(int)
        n = D.shape[0]
        man = pywrapcp.RoutingIndexManager(n, 1, 0)
        rm = pywrapcp.RoutingModel(man)
        cb = rm.RegisterTransitCallback(lambda i, j: int(D[man.IndexToNode(i), man.IndexToNode(j)]))
        rm.SetArcCostEvaluatorOfAllVehicles(cb)
        p = pywrapcp.DefaultRoutingSearchParameters()
        p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        p.time_limit.seconds = secs
        sol = rm.SolveWithParameters(p)
        res[name] = sol.ObjectiveValue()
        print(name, res[name], OPT[name], flush=True)
    json.dump(res, open(f"{out}/ortools_{secs}s.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--explore", default="mae")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--secs", type=int, default=10)
    ap.add_argument("--out", default="results/tsp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "train":
        train(a.seed, a.explore, a.episodes, a.out)
    elif a.cmd == "eval":
        evaluate(a.out, a.runs, [2000, 8000], list(OPT), a.models)
    elif a.cmd == "ortools":
        ortools_ref(a.out, list(OPT), a.secs)
