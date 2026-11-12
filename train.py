"""Train a tabular Q-learning or SARSA agent on EVChargingEnv.

Example:
    python3 train.py --algo qlearning --episodes 100000 --seed 0
    python3 train.py --algo sarsa     --episodes 100000 --seed 0

Saves to results/:
    <algo>_seed<k>_Q.npy    learned Q-table (1008 x 3)
    <algo>_seed<k>_log.npz  per-episode training logs + periodic greedy evaluations
"""
import argparse
import os
import time

import numpy as np

from agents import AGENTS
from ev_env import EVChargingEnv, state_index

TRAIN_SEED0 = 0            # training nights use seeds 0, 1, 2, ...
VAL_SEED0 = 900_000        # fixed validation nights for the learning curve
MISS_TOL = 0.5             # same "miss" definition as baselines.py (% points)


def linear(start, end, frac_done, decay_frac):
    """Linear decay from start to end over the first decay_frac of training."""
    return end + (start - end) * max(0.0, 1.0 - frac_done / decay_frac)


def run_greedy(agent, env, seeds):
    """Greedy evaluation (epsilon = 0) on fixed seeds -> mean return, cost, miss rate.
    The agent's RNG state is restored afterwards so evaluating doesn't change training."""
    rng_state = agent.rng.bit_generator.state
    rets, costs, misses = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done, total = False, 0.0
        while not done:
            obs, r, done, _, info = env.step(agent.act(state_index(obs), greedy=True))
            total += r
        rets.append(total); costs.append(info["total_cost"]); misses.append(info["shortfall"] > MISS_TOL)
    agent.rng.bit_generator.state = rng_state
    return np.mean(rets), np.mean(costs), np.mean(misses)


def train(algo="qlearning", episodes=100_000, alpha=0.1, alpha_end=0.01, gamma=1.0,
          eps_start=1.0, eps_end=0.05, decay_frac=0.8, seed=0, eval_every=2_000,
          n_val=500, env_kwargs=None, verbose=True):
    env = EVChargingEnv(**(env_kwargs or {}))
    agent = AGENTS[algo](alpha=alpha, gamma=gamma, epsilon=eps_start, seed=seed)
    val_seeds = VAL_SEED0 + np.arange(n_val)
    ep_return, ep_cost, ep_miss = (np.zeros(episodes) for _ in range(3))
    evals = []
    t0 = time.time()

    for ep in range(episodes):
        frac = ep / episodes
        agent.epsilon = linear(eps_start, eps_end, frac, decay_frac)
        agent.alpha = linear(alpha, alpha_end, frac, 1.0)

        obs, _ = env.reset(seed=TRAIN_SEED0 + seed * episodes + ep)
        s = state_index(obs)
        a = agent.act(s)
        done, total = False, 0.0
        while not done:
            obs, r, done, _, info = env.step(a)          # interact with the environment
            s2 = state_index(obs)
            a2 = None if done else agent.act(s2)         # next action (used by SARSA)
            agent.update(s, a, r, s2, a2, done)          # TD update of Q(s, a)
            s, a = s2, a2
            total += r

        ep_return[ep], ep_cost[ep], ep_miss[ep] = total, info["total_cost"], info["shortfall"] > MISS_TOL
        if (ep + 1) % eval_every == 0:
            evals.append((ep + 1, *run_greedy(agent, env, val_seeds)))
            if verbose:
                e = evals[-1]
                print(f"[{algo} s{seed}] ep {ep+1:>6}  eps={agent.epsilon:.2f}  "
                      f"val return={e[1]:7.2f}  cost=${e[2]:.2f}  miss={e[3]*100:4.1f}%  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    log = dict(ep_return=ep_return, ep_cost=ep_cost, ep_miss=ep_miss, evals=np.array(evals),
               config=np.array([algo, episodes, alpha, alpha_end, gamma, eps_start, eps_end,
                                decay_frac, seed], dtype=object))
    return agent, log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(AGENTS), default="qlearning")
    p.add_argument("--episodes", type=int, default=100_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--alpha-end", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--decay-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    agent, log = train(args.algo, args.episodes, args.alpha, args.alpha_end, args.gamma,
                       args.eps_start, args.eps_end, args.decay_frac, args.seed, args.eval_every)
    tag = os.path.join(args.out, f"{args.algo}_seed{args.seed}")
    agent.save(tag + "_Q.npy")
    np.savez(tag + "_log.npz", **log)
    print("saved", tag + "_Q.npy", "and", tag + "_log.npz")
