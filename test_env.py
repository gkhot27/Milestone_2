"""Sanity tests for EVChargingEnv. Run:  python3 test_env.py"""
import numpy as np
from gymnasium.utils.env_checker import check_env
from ev_env import EVChargingEnv, state_index, OFF, SLOW, FAST, PRICE_TRANSITIONS
from baselines import oracle_plan, run_episode, charge_immediately, time_of_use

def ok(msg): print(f"[PASS] {msg}")

env = EVChargingEnv()

# 1. Gymnasium API compliance
check_env(EVChargingEnv(), skip_render_check=True)
ok("gymnasium check_env: spaces, reset/step signatures, seeding")

# 2. reset() initializes within the specified ranges
hs, bs, plug, dep, early = [], [], [], [], 0
N = 10000
for s in range(N):
    obs, info = env.reset(seed=s)
    assert env.observation_space.contains(obs)
    hs.append(obs[0]); bs.append(info["battery"]); plug.append(env.plug_in)
    dep.append(info["planned_departure"]); early += info["early_departure"]
    assert env.total_cost == 0 and env.fast_hours == 0 and env.history == []
assert set(plug) == {17, 18, 19, 20, 21} and set(dep) == {6, 7, 8}
assert 0.20 <= min(bs) and max(bs) <= 0.50 and 9 <= min(hs) and max(hs) <= 15
ok(f"reset(): plug-in 17-21h, departure 6-8am, battery {min(bs)*100:.1f}-{max(bs)*100:.1f}%, "
   f"h0 in [{min(hs)},{max(hs)}], counters zeroed")
ok(f"early departure rate = {early / N * 100:.1f}% (expected ~10%)")

# 3. Same seed -> identical episode; different seed -> different
o1, _ = env.reset(seed=42); p1 = env.price_path.copy()
o2, _ = env.reset(seed=42); p2 = env.price_path.copy()
o3, _ = env.reset(seed=43); p3 = env.price_path.copy()
assert np.array_equal(o1, o2) and np.array_equal(p1, p2)
assert not (np.array_equal(o1, o3) and np.array_equal(p1, p3))
ok("reset(seed) is reproducible")

# 4. Action effects on battery, cost and reward
for a, kw in ((OFF, 0.0), (SLOW, 3.7), (FAST, 7.4)):
    obs, info = env.reset(seed=7)
    b0, price = info["battery"], info["price_per_kwh"]
    obs, r, term, trunc, info = env.step(a)
    assert np.isclose(info["battery"] - b0, kw / 60)
    assert np.isclose(r, -price * kw - env.lam * (a == FAST))
    assert obs[0] == env.planned_departure - env.clock
ok("step(): off +0%, slow +6.17%, fast +12.33%; reward = -price*kWh - lambda*1[fast]; h decrements")

# 5. Battery never exceeds 100%
env.reset(seed=1); env.battery = 0.97
_, r, *_ , info = env.step(FAST)
assert info["battery"] == 1.0 and np.isclose(info["energy"], 0.03 * 60)
ok("battery capped at 100% (only 1.8 kWh billed when 97% -> 100%)")

# 6. Termination at ACTUAL departure + terminal penalty
for s in range(500):
    obs, info = env.reset(seed=s)
    steps, done = 0, False
    while not done:
        obs, r, done, _, info = env.step(OFF); steps += 1
    assert steps == env.actual_departure - env.plug_in
    exp_short = (env.target - info["battery"]) * 100
    assert np.isclose(info["shortfall"], exp_short)
    assert np.isclose(r, -env.M * exp_short)
ok("episode ends at actual departure (incl. early exits); final reward includes -M*shortfall")

# 7. Price Markov chain: evening expensive, night cheap
counts = {k: np.zeros(3) for k in PRICE_TRANSITIONS}
from ev_env import period_of
for s in range(3000):
    env.reset(seed=s)
    for i, p in enumerate(env.price_path):
        counts[period_of((env.plug_in + i) % 24)][p] += 1
fr = {k: v / v.sum() for k, v in counts.items()}
assert fr["evening"][2] > fr["night"][2] and fr["night"][0] > fr["evening"][0]
ok("price chain: P(high) evening={:.2f} vs night={:.2f};  P(low) night={:.2f} vs evening={:.2f}".format(
    fr["evening"][2], fr["night"][2], fr["night"][0], fr["evening"][0]))

# 8. State indexing covers exactly 16*21*3 = 1008 states
idx = {state_index((h, b, p)) for h in range(16) for b in range(21) for p in range(3)}
assert idx == set(range(1008))
ok("state_index maps (h,b,p) one-to-one onto 0..1007 for the Q-table")

# 9. Oracle DP value matches the return actually obtained in the env
for s in range(200):
    env.reset(seed=s); plan, v = oracle_plan(env)
    total = 0
    for a in plan:
        obs, r, done, _, info = env.step(a)
        total += r
    assert np.isclose(total, v), (s, total, v)
ok("oracle DP value == realized env return on 200 episodes")
print("\nAll environment tests passed.")
