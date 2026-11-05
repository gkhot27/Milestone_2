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
├── test_env.py          # 10 sanity checks for the environment
├── make_figures.py      # Generates report figures into figures/
├── baseline_results.txt # Saved output from baselines.py
├── test_output.txt      # Saved output from test_env.py
├── figures/             # Generated PNG figures
└── report/              # Progress report (Khot_Gururaj_milestone2.pdf)
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
pip install numpy gymnasium matplotlib pygments
```

---

## Execution Instructions

All commands are run from the project root. Make sure your virtual
environment is activated first (`source .venv/bin/activate`).

### 1. Run the environment tests

Runs 10 sanity checks over the environment and the oracle, and saves the output
to `test_output.txt` (used by `make_figures.py`).

```bash
python3 test_env.py | tee test_output.txt
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

Evaluates all three baselines over 5,000 episodes each, and saves the output
to `baseline_results.txt` (used by `make_figures.py`).

```bash
python3 baselines.py | tee baseline_results.txt
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

### 3. Generate the report figures

Creates `fig1_episodes.png`, `fig2_step_code.png`, and `fig3_test_output.png`
in the `figures/` directory. Requires Matplotlib and Pygments.

```bash
python3 make_figures.py
```

The script reads `test_output.txt` and `baseline_results.txt`, so run steps 1
and 2 first (with `| tee ...`) to make sure those files reflect the current code.
The scripts only print to the terminal, so running them without `tee` does not
update the files.

### 4. Read the report

The compiled progress report is `report/Khot_Gururaj_milestone2.pdf`. Its figures are
the ones produced in step 3.

---

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

Completed so far: the environment, three baselines, and the test suite. The
remaining work (Q-learning, SARSA, learning curves, λ sweep, early-departure
comparison, and Q-table heatmaps) is outlined in the progress report.
