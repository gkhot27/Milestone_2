"""Tabular Q-learning and SARSA agents for EVChargingEnv.

Both agents keep a Q-table of shape (1008 states, 3 actions) indexed with
ev_env.state_index(obs). They share epsilon-greedy action selection and differ
only in the TD target used by update().
"""
import numpy as np

from ev_env import state_index  # noqa: F401  (re-exported for convenience)

N_STATES, N_ACTIONS = 16 * 21 * 3, 3


class TabularAgent:
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=1.0, seed=0):
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        """Epsilon-greedy: random action with prob. epsilon, else the best Q-value
        (ties broken at random so an all-zero row isn't always 'off')."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, a2, done):
        target = r if done else r + self.gamma * self.next_value(s2, a2)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def next_value(self, s2, a2):
        raise NotImplementedError

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
        return self


class QLearningAgent(TabularAgent):
    """Off-policy: bootstraps from the best next action, max_a' Q(s', a')."""
    def next_value(self, s2, a2):
        return self.Q[s2].max()


class SarsaAgent(TabularAgent):
    """On-policy: bootstraps from the action it will actually take, Q(s', a')."""
    def next_value(self, s2, a2):
        return self.Q[s2, a2]


AGENTS = {"qlearning": QLearningAgent, "sarsa": SarsaAgent}
