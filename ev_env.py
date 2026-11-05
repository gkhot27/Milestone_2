"""
Smart EV Charging Under Price Uncertainty -- custom Gymnasium environment.

One episode = one night. The car is plugged in between 5pm and 9pm, and the
agent picks a charge rate every hour until the driver leaves (6-8am, with a
10% chance of leaving 1-2 hours early).

Observation  s_t = (h_t, b_t, p_t)
    h_t : hours left until the *planned* departure        (0..15)
    b_t : battery level rounded to nearest 5%  (index 0..20 -> 0%..100%)
    p_t : price level  (0 = low, 1 = medium, 2 = high)
Action       a_t in {0: off, 1: slow 3.7 kW, 2: fast 7.4 kW}
Reward       r_t = -c(p_t) * e_t - lambda * 1[a_t = fast]
             at departure: additional -M * max(0, b* - b_T)   (b in % points)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

OFF, SLOW, FAST = 0, 1, 2
LOW, MED, HIGH = 0, 1, 2

# Time-of-day dependent price Markov chain. Rows = current level, cols = next.
#                     low   med   high
PRICE_TRANSITIONS = {
    "evening":  np.array([[0.50, 0.40, 0.10],     # 5pm - 9pm  (peak)
                          [0.10, 0.50, 0.40],
                          [0.05, 0.25, 0.70]]),
    "shoulder": np.array([[0.60, 0.35, 0.05],     # 10pm - 11pm
                          [0.30, 0.55, 0.15],
                          [0.15, 0.50, 0.35]]),
    "night":    np.array([[0.85, 0.13, 0.02],     # midnight - 4am (cheap)
                          [0.55, 0.40, 0.05],
                          [0.40, 0.40, 0.20]]),
    "morning":  np.array([[0.55, 0.40, 0.05],     # 5am - 8am
                          [0.20, 0.60, 0.20],
                          [0.10, 0.40, 0.50]]),
}
INITIAL_PRICE_DIST = np.array([0.10, 0.40, 0.50])   # price level at plug-in


def period_of(hour_of_day: int) -> str:
    if 17 <= hour_of_day <= 21:
        return "evening"
    if hour_of_day in (22, 23):
        return "shoulder"
    if 0 <= hour_of_day <= 4:
        return "night"
    return "morning"


class EVChargingEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, capacity_kwh=60.0, target=0.80, wear_penalty=0.10,
                 miss_penalty=5.0, prices=(0.08, 0.15, 0.32),
                 early_departure_prob=0.10, volatility=0.0, render_mode=None):
        super().__init__()
        self.capacity = capacity_kwh
        self.target = target                    # b*  (fraction)
        self.lam = wear_penalty                 # lambda, $ per fast-charging hour
        self.M = miss_penalty                   # $ per % point below target
        self.prices = np.array(prices)          # $/kWh for low/med/high
        self.rates_kw = np.array([0.0, 3.7, 7.4])
        self.early_departure_prob = early_departure_prob
        self.volatility = volatility            # 0 = normal, >0 shifts mass toward HIGH
        self.render_mode = render_mode

        self.observation_space = spaces.MultiDiscrete([16, 21, 3])
        self.action_space = spaces.Discrete(3)

    # ------------------------------------------------------------------ helpers
    def _transition_matrix(self, hour_of_day):
        P = PRICE_TRANSITIONS[period_of(hour_of_day)].copy()
        if self.volatility > 0:                 # used for the generalization test
            P = (1 - self.volatility) * P + self.volatility * np.array([0, 0, 1.0])
        return P

    def _sample_price_path(self):
        """Pre-sample the price level for every hour of the night. Prices do not
        depend on the agent's actions, so this is equivalent to sampling online,
        and it lets the oracle baseline see the future prices."""
        n_hours = self.planned_departure - self.plug_in
        path = np.empty(n_hours, dtype=int)
        path[0] = self.np_random.choice(3, p=INITIAL_PRICE_DIST)
        for i in range(1, n_hours):
            hod = (self.plug_in + i) % 24
            path[i] = self.np_random.choice(3, p=self._transition_matrix(hod)[path[i - 1]])
        return path

    def _obs(self):
        h = self.planned_departure - self.clock
        b = int(np.clip(np.rint(self.battery * 20), 0, 20))
        p = int(self.price_path[min(self.clock - self.plug_in, len(self.price_path) - 1)])
        return np.array([h, b, p], dtype=np.int64)

    def _info(self, **extra):
        idx = min(self.clock - self.plug_in, len(self.price_path) - 1)
        return {"battery": self.battery, "hour_of_day": self.clock % 24,
                "price_per_kwh": float(self.prices[self.price_path[idx]]),
                "planned_departure": self.planned_departure % 24,
                "actual_departure": self.actual_departure % 24,
                "early_departure": self.actual_departure < self.planned_departure,
                "total_cost": self.total_cost, "fast_hours": self.fast_hours, **extra}

    # --------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.plug_in = int(rng.integers(17, 22))              # 5pm..9pm
        self.planned_departure = 24 + int(rng.integers(6, 9)) # 6am..8am next day
        self.actual_departure = self.planned_departure
        if rng.random() < self.early_departure_prob:          # surprise early exit
            self.actual_departure -= int(rng.integers(1, 3))  # 1 or 2 hours early
        self.battery = float(rng.uniform(0.20, 0.50))
        self.clock = self.plug_in
        self.price_path = self._sample_price_path()
        self.total_cost, self.fast_hours, self.history = 0.0, 0, []
        return self._obs(), self._info()

    def step(self, action):
        assert self.action_space.contains(action), f"invalid action {action}"
        price = self.prices[self.price_path[self.clock - self.plug_in]]

        # Transition: add energy (1-hour step), capped at 100% battery
        energy = min(self.rates_kw[action] * 1.0, (1.0 - self.battery) * self.capacity)
        self.battery = min(1.0, self.battery + energy / self.capacity)

        # Reward: electricity cost + battery-wear penalty for fast charging
        cost = price * energy
        reward = -cost - self.lam * (action == FAST)
        self.total_cost += cost
        self.fast_hours += int(action == FAST)

        self.clock += 1
        terminated = self.clock >= self.actual_departure
        shortfall = 0.0
        if terminated:                      # driver leaves: penalize missed target
            shortfall = max(0.0, self.target - self.battery) * 100   # % points
            if shortfall < 1e-6:
                shortfall = 0.0
            reward -= self.M * shortfall

        self.history.append((self.clock - 1, int(action), float(price), self.battery, -cost))
        if self.render_mode == "human":
            print(self.render())
        return (self._obs(), float(reward), terminated, False,
                self._info(energy=energy, cost=cost, shortfall=shortfall))

    def render(self):
        hod, a, price, batt, _ = self.history[-1] if self.history else (self.clock, 0, 0, self.battery, 0)
        return (f"{hod % 24:02d}:00  action={['off ', 'slow', 'fast'][a]}  "
                f"price=${price:.2f}/kWh  battery={batt * 100:5.1f}%")


def state_index(obs):
    """Flatten (h, b, p) into a single index 0..1007 for a Q-table."""
    h, b, p = obs
    return int((h * 21 + b) * 3 + p)
