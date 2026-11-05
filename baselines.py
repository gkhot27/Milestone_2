"""Rule-based baselines and the perfect-information oracle (dynamic programming)."""
import numpy as np
from ev_env import EVChargingEnv, OFF, SLOW, FAST

OFF_PEAK = {23, 0, 1, 2, 3, 4, 5}
MISS_TOL = 0.5   # count a "miss" if the car leaves >0.5 % points below target


def charge_immediately(obs, info, env):
    """Default car behaviour: fast charge from plug-in until the target is reached."""
    return FAST if info["battery"] < env.target - 1e-9 else OFF


def time_of_use(obs, info, env):
    """Charge slowly during fixed off-peak hours; switch to fast if the (planned)
    deadline is close enough that waiting another hour would make the target
    unreachable."""
    needed = env.target - info["battery"]
    if needed <= 1e-9:
        return OFF
    h_left = obs[0]
    fast_pct = env.rates_kw[FAST] / env.capacity
    if needed > (h_left - 1) * fast_pct:
        return FAST
    return SLOW if info["hour_of_day"] in OFF_PEAK else OFF


def oracle_plan(env):
    """Backward DP with full knowledge of the price path and the ACTUAL departure.
    Battery is tracked exactly in units of 3.7 kWh (slow = 1 unit, fast = 2)."""
    e = env.unwrapped
    unit = e.rates_kw[SLOW]
    T = e.actual_departure - e.clock
    b0 = e.battery
    k_max = int(np.ceil((1 - b0) * e.capacity / unit)) + 1
    batt = lambda k: min(1.0, b0 + k * unit / e.capacity)

    V = np.array([-e.M * max(0.0, (e.target - batt(k)) * 100) if (e.target - batt(k)) > 1e-6 else 0.0
                  for k in range(k_max + 1)])
    policy = np.zeros((T, k_max + 1), dtype=int)
    for t in reversed(range(T)):
        price = e.prices[e.price_path[e.clock - e.plug_in + t]]
        newV = np.full(k_max + 1, -np.inf)
        for k in range(k_max + 1):
            for a, units in ((OFF, 0), (SLOW, 1), (FAST, 2)):
                energy = min(units * unit, (1 - batt(k)) * e.capacity)
                q = -price * energy - e.lam * (a == FAST) + V[min(k + units, k_max)]
                if q > newV[k] + 1e-12:
                    newV[k], policy[t, k] = q, a
        V = newV
    # roll the plan forward
    plan, k = [], 0
    for t in range(T):
        a = policy[t, k]
        plan.append(a)
        k = min(k + (0, 1, 2)[a], k_max)
    return plan, V[0]


def run_episode(env, policy, seed=None):
    obs, info = env.reset(seed=seed)
    plan = oracle_plan(env)[0] if policy == "oracle" else None
    total_r, t, done = 0.0, 0, False
    while not done:
        a = plan[t] if plan is not None else policy(obs, info, env)
        obs, r, done, _, info = env.step(a)
        total_r += r
        t += 1
    return {"return": total_r, "cost": info["total_cost"], "missed": info["shortfall"] > MISS_TOL,
            "shortfall": info["shortfall"], "fast_hours": info["fast_hours"],
            "early": info["early_departure"]}


def evaluate(policy, n=5000, seed=0, **env_kwargs):
    env = EVChargingEnv(**env_kwargs)
    rs = [run_episode(env, policy, seed=seed + i) for i in range(n)]
    agg = lambda k: float(np.mean([r[k] for r in rs]))
    return {"return": agg("return"), "cost": agg("cost"), "miss_rate": agg("missed"),
            "fast_hours": agg("fast_hours"), "shortfall": agg("shortfall")}


if __name__ == "__main__":
    print(f"{'policy':<20}{'avg cost ($)':>14}{'miss rate':>12}{'fast hrs':>10}{'avg return':>12}")
    for name, pol in [("charge immediately", charge_immediately),
                      ("time-of-use rule", time_of_use), ("oracle (DP)", "oracle")]:
        m = evaluate(pol)
        print(f"{name:<20}{m['cost']:>14.2f}{m['miss_rate'] * 100:>11.1f}%"
              f"{m['fast_hours']:>10.2f}{m['return']:>12.2f}")
