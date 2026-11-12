# Smart EV Charging Under Price Uncertainty

A reinforcement-learning project that controls an EV charger overnight. The
agent picks **off**, **slow (3.7 kW)**, or **fast (7.4 kW)** every hour with the
goal of reaching 80% battery before the driver leaves, while keeping the
electricity cost low. Prices (low / medium / high) and the actual departure
time are random, so the agent must plan under uncertainty.

One episode = one night. The car has a 60 kWh battery and the price is one of
$0.08, $0.15, or $0.32 per kWh. There is a 10% chance the driver leaves 1-2
hours early, which the agent cannot observe in its state.

---

## Project Structure

```
Smart-EV-charging/
├── ev_env.py            # Custom Gymnasium environment (the simulation)
├── baselines.py         # Three baselines: charge-immediately, time-of-use, oracle (DP)
├── agents.py            # Tabular Q-learning and SARSA agents        (Milestone 3)
├── train.py             # Training loop + periodic greedy validation (Milestone 3)
├── evaluate.py          # Test-set comparison of agents vs. baselines (Milestone 3)
├── plot_results.py      # Milestone 3 figures                         (Milestone 3)
├── test_env.py          # 10 sanity checks for the environment
├── make_figures.py      # Milestone 2 figures
├── requirements.txt     # Python dependencies
├── baseline_results.txt # Saved output from baselines.py
├── test_output.txt      # Saved output from test_env.py
├── results/             # Learned Q-tables, training logs, eval_results.csv/.txt
├── logs/                # Console output of each training run
├── figures/             # Milestone 2 figures; figures/milestone3/ for Milestone 3
└── report/              # Milestone report PDFs
```

### Core Files

- **`ev_env.py`** — `EVChargingEnv`, a custom [Gymnasium](https://gymnasium.farama.org/)
  environment. Defines the state space `(hours_left, battery_bin, price_level)`
  with `16 × 21 × 3 = 1008` discrete states, a 3-action space, the reward
  function, the price Markov chain, and early-departure logic.

- **`baselines.py`** — Three policies and the evaluation harness:
  - `charge_immediately` — fast charge until the target is reached (default car behavior).
  - `time_of_use` — slow charge during off-peak hours (11pm-5am), fast only if the deadline is tight.
  - `oracle_plan` — backward dynamic programming with full knowledge of the price path and actual departure time (upper bound).
  - `evaluate()` — runs N episodes and reports average cost, miss rate, fast hours, and return.

- **`test_env.py`** — 10 assert-based tests: Gymnasium API compliance, reset ranges,
  seeding reproducibility, action effects, battery capping, termination logic,
  price-chain statistics, state indexing, and oracle value verification.

- **`make_figures.py`** — Produces `fig1_episodes.png` (two example nights with all
  baselines), `fig2_step_code.png` (the `step()` source), and
  `fig3_test_output.png` (test + baseline results).

- **`agents.py`** — `QLearningAgent` and `SarsaAgent`. Both keep a `1008 × 3`
  Q-table, pick actions epsilon-greedily (random tie-breaking), and do a one-step
  TD update. The only difference is the bootstrap target: `max_a' Q(s', a')`
  for Q-learning vs. `Q(s', a')` for the action SARSA actually takes next.

- **`train.py`** — trains one agent for one seed. Epsilon decays linearly from
  1.0 to 0.05 over the first 80% of episodes and alpha from 0.1 to 0.01. Every
  2,000 episodes the greedy policy is evaluated on 500 fixed validation nights
  (this does not change the training run).

- **`evaluate.py`** — runs every policy (random, charge-immediately,
  time-of-use, Q-learning, SARSA, oracle) on the same 5,000 held-out test
  nights and writes `results/eval_results.csv`.

- **`plot_results.py`** — Milestone 3 figures: agent code, training loop,
  learning curves, and the baseline comparison.

---

## Dependencies

- Python 3.9+
- [NumPy](https://numpy.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Matplotlib](https://matplotlib.org/) (for figure generation only)
- [Pygments](https://pygments.org/) (for figure generation only)

---

## Setup

Create a virtual environment and install the dependencies:

```bash
cd Smart-EV-charging
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Execution Instructions

All commands are run from the project root. Make sure your virtual
environment is activated first (`source .venv/bin/activate`).

### 1. Run the environment tests

Runs 10 sanity checks over the environment and the oracle.

```bash
python3 test_env.py
```

Expected output (see `test_output.txt` for the saved version):

```
[PASS] gymnasium check_env: spaces, reset/step signatures, seeding
[PASS] reset(): plug-in 17-21h, departure 6-8am, battery 20.0-50.0%, h0 in [9,15], counters zeroed
[PASS] early departure rate = 10.1% (expected ~10%)
[PASS] reset(seed) is reproducible
[PASS] step(): off +0%, slow +6.17%, fast +12.33%; reward = -price*kWh - lambda*1[fast]; h decrements
[PASS] battery capped at 100% (only 1.8 kWh billed when 97% -> 100%)
[PASS] episode ends at actual departure (incl. early exits); final reward includes -M*shortfall
[PASS] price chain: P(high) evening=0.52 vs night=0.04;  P(low) night=0.74 vs evening=0.11
[PASS] state_index maps (h,b,p) one-to-one onto 0..1007 for the Q-table
[PASS] oracle DP value == realized env return on 200 episodes

All environment tests passed.
```

### 2. Run the baselines

Evaluates all three baselines over 5,000 episodes each.

```bash
python3 baselines.py
```

Expected output (see `baseline_results.txt`):

```
policy                avg cost ($)   miss rate  fast hrs  avg return
charge immediately            6.23        0.0%      4.11       -6.64
time-of-use rule              3.61        6.6%      1.11       -6.34
oracle (DP)                   2.47        0.0%      2.02       -2.68
```

The oracle achieves the lowest cost ($2.47) with zero misses — this is the
upper bound that the RL agents (Q-learning and SARSA) will aim to close.


### 3. Train the RL agents (Milestone 3)

Each run takes about 1 minute on a laptop. The report uses 3 seeds per algorithm:

```bash
for algo in qlearning sarsa; do
  for seed in 0 1 2; do
    python3 train.py --algo $algo --seed $seed | tee logs/train_${algo}_seed${seed}.txt
  done
done
```

Each run saves `results/<algo>_seed<k>_Q.npy` (the Q-table) and
`results/<algo>_seed<k>_log.npz` (per-episode returns + validation curve).
Useful flags: `--episodes` (default 100000), `--alpha` / `--alpha-end`
(0.1 → 0.01), `--gamma` (1.0), `--eps-start` / `--eps-end` (1.0 → 0.05),
`--decay-frac` (0.8). Runs are fully reproducible from `--seed`.

The trained Q-tables are already in `results/`, so you can skip straight to
step 4.

### 4. Evaluate against the baselines

```bash
python3 evaluate.py | tee results/eval_results.txt
```

All policies are tested on the same 5,000 nights (seeds 1,000,000+), which are
never used in training (seeds 0–299,999) or validation (900,000+):

```
policy                 avg cost ($)  miss rate  miss (early)  fast hrs  avg return
Random policy                  5.70       9.9%         16.0%      3.92      -11.03
Charge immediately             6.28       0.0%          0.0%      4.14       -6.69
Time-of-use rule               3.65       6.9%         67.6%      1.15       -6.30
Q-learning (3 seeds)           2.85       0.5%          3.4%      3.06       -3.43
SARSA (3 seeds)                3.36       0.6%          1.5%      1.96       -4.32
Oracle (DP)                    2.48       0.0%          0.0%      2.06       -2.69
```

"miss (early)" is the miss rate on just the nights where the driver left early.

### 5. Make the Milestone 3 figures

```bash
python3 plot_results.py      # -> figures/milestone3/
```

## RL Hyperparameters

| Parameter | Value |
|---|---|
| Episodes per run | 100,000 (one night each) |
| Learning rate α | 0.1 → 0.01 (linear decay) |
| Discount γ | 1.0 (episodic; the objective is total cost per night) |
| Exploration ε | 1.0 → 0.05 over the first 80% of episodes, then 0.05 |
| Q-table init | zeros (optimistic, since all rewards are ≤ 0) |
| Seeds | 3 per algorithm |

## Key Design Parameters

| Parameter | Value | Description |
|---|---|---|
| Battery capacity | 60 kWh | EV battery size |
| Target charge | 80% | Goal the agent must reach before departure |
| Prices | $0.08 / $0.15 / $0.32 per kWh | Low / medium / high price levels |
| Wear penalty (λ) | $0.10 per fast hour | Discourages fast charging |
| Miss penalty (M) | $5.00 per % point below target | Terminal penalty for missing 80% |
| Early departure | 10% chance, 1-2 hours early | Hidden from the agent's state |
| Plug-in time | 5pm-9pm (random) | Start of the episode |
| Departure time | 6am-8am (random) | Planned end of the episode |

## State and Action Spaces

- **State** `s = (h, b, p)`:
  - `h` — hours left until the planned departure (0-15)
  - `b` — battery level rounded to the nearest 5% (21 bins, 0%-100%)
  - `p` — price level (0 = low, 1 = medium, 2 = high)
  - Total: `16 × 21 × 3 = 1008` discrete states

- **Action** `a ∈ {0, 1, 2}`: off, slow (3.7 kW ≈ +6%/hr), fast (7.4 kW ≈ +12%/hr)

- **Reward**: `r_t = -price · energy - λ · 1[a=fast]`, plus a terminal penalty
  `-M · max(0, 80% - b_final)` in percentage points when the driver leaves.

## Status

- **Milestone 2:** environment, three baselines, and the test suite.
- **Milestone 3:** Q-learning and SARSA implemented, trained (3 seeds each), and
  evaluated. Both beat every baseline on cost while staying under a 1% miss rate.
- **Next:** λ sweep (cost vs. battery wear), early departures on/off,
  Q-table heatmaps, and the volatile-price generalization test.
