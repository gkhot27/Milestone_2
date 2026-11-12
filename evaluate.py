"""Evaluate trained agents against the baselines on held-out test nights.

    python3 evaluate.py                 # uses results/<algo>_seed*_Q.npy
    python3 evaluate.py --n 5000

All policies see the SAME 5,000 test nights (seeds 1,000,000+), which were never
used for training (0..299,999) or validation (900,000+). Agents act greedily.
Writes results/eval_results.csv and prints a table.
"""
import argparse
import csv
import glob
import os

import numpy as np

from agents import AGENTS
from baselines import charge_immediately, time_of_use, run_episode
from ev_env import EVChargingEnv, state_index

TEST_SEED0 = 1_000_000


def greedy_policy(agent):
    return lambda obs, info, env: agent.act(state_index(obs), greedy=True)


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda obs, info, env: int(rng.integers(3))


def evaluate_policy(policy, n, env_kwargs=None):
    env = EVChargingEnv(**(env_kwargs or {}))
    rs = [run_episode(env, policy, seed=TEST_SEED0 + i) for i in range(n)]
    early = [r for r in rs if r["early"]]
    return {"cost": np.mean([r["cost"] for r in rs]),
            "miss_rate": np.mean([r["missed"] for r in rs]) * 100,
            "miss_rate_early": np.mean([r["missed"] for r in early]) * 100,
            "fast_hours": np.mean([r["fast_hours"] for r in rs]),
            "return": np.mean([r["return"] for r in rs])}


def main(n=5000, results="results"):
    rows = [("Random policy", evaluate_policy(random_policy(), n)),
            ("Charge immediately", evaluate_policy(charge_immediately, n)),
            ("Time-of-use rule", evaluate_policy(time_of_use, n))]
    for algo, name in (("qlearning", "Q-learning"), ("sarsa", "SARSA")):
        per_seed = []
        for path in sorted(glob.glob(os.path.join(results, f"{algo}_seed*_Q.npy"))):
            agent = AGENTS[algo]().load(path)
            per_seed.append(evaluate_policy(greedy_policy(agent), n))
        if per_seed:   # average over training seeds, keep the spread of cost/miss rate
            m = {k: np.mean([p[k] for p in per_seed]) for k in per_seed[0]}
            m["cost_std"] = np.std([p["cost"] for p in per_seed])
            m["miss_std"] = np.std([p["miss_rate"] for p in per_seed])
            rows.append((f"{name} ({len(per_seed)} seeds)", m))
    rows.append(("Oracle (DP)", evaluate_policy("oracle", n)))

    hdr = f"{'policy':<22}{'avg cost ($)':>13}{'miss rate':>11}{'miss (early)':>14}{'fast hrs':>10}{'avg return':>12}"
    print(hdr)
    for name, m in rows:
        print(f"{name:<22}{m['cost']:>13.2f}{m['miss_rate']:>10.1f}%{m['miss_rate_early']:>13.1f}%"
              f"{m['fast_hours']:>10.2f}{m['return']:>12.2f}")
    with open(os.path.join(results, "eval_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "cost", "cost_std", "miss_rate", "miss_std", "miss_rate_early",
                    "fast_hours", "return"])
        for name, m in rows:
            w.writerow([name, f"{m['cost']:.4f}", f"{m.get('cost_std', 0):.4f}", f"{m['miss_rate']:.3f}",
                        f"{m.get('miss_std', 0):.3f}", f"{m['miss_rate_early']:.3f}",
                        f"{m['fast_hours']:.3f}", f"{m['return']:.4f}"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--results", default="results")
    a = p.parse_args()
    main(a.n, a.results)
