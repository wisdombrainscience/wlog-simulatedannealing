import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sadrl.dqn import run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--env", required=True)
p.add_argument("--method", required=True)
p.add_argument("--seeds", default="0")
p.add_argument("--steps", type=int, default=None)
p.add_argument("--out", default="results/rl")
p.add_argument("--tag", default="")
p.add_argument("--override", default="{}")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
for sd in [int(x) for x in a.seeds.split(",")]:
    fn = f"{a.out}/{a.env}__{a.method}{a.tag}__s{sd}.json"
    if os.path.exists(fn):
        continue
    log = run(a.env, a.method, sd, a.steps, overrides=json.loads(a.override))
    with open(fn, "w") as f:
        json.dump(log, f)
    print(fn, f"{log['wall']:.0f}s final={log['eval_ret'][-1]:.2f}", flush=True)
