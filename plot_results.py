"""Milestone 3 figures -> figures/milestone3/

fig1_agent_code.png       Q-learning / SARSA implementation (from agents.py)
fig2_training_loop.png    the training loop (from train.py)
fig3_learning_curves.png  greedy validation return / miss rate / cost vs. training episodes
fig4_baseline_comparison.png  test-set cost and miss rate vs. the baselines

Run after train.py (all seeds) and evaluate.py.
"""
import csv
import glob
import inspect
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import agents
import train
from baselines import run_episode, time_of_use
from ev_env import EVChargingEnv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "milestone3")
RES = os.path.join(HERE, "results")
COLORS = {"qlearning": "#2E6FBA", "sarsa": "#D9822B"}
NAMES = {"qlearning": "Q-learning", "sarsa": "SARSA"}


def code_png(src, path):
    """Syntax-highlighted code screenshot (same style as Milestone 2's Figure 2)."""
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import PythonLexer
    kw = dict(line_numbers=True, font_size=22, line_number_bg="#E8EAD8",
              line_number_fg="#888888", image_pad=18, style="default")
    for font in ("Menlo", "DejaVu Sans Mono", "Consolas", "Courier New"):
        try:
            png = highlight(src, PythonLexer(), ImageFormatter(font_name=font, **kw))
            break
        except Exception:
            continue
    with open(path, "wb") as f:
        f.write(png)


def agent_source():
    parts = [inspect.getsource(agents.TabularAgent.act), inspect.getsource(agents.TabularAgent.update),
             "class QLearningAgent(TabularAgent):\n" + inspect.getsource(agents.QLearningAgent.next_value),
             "class SarsaAgent(TabularAgent):\n" + inspect.getsource(agents.SarsaAgent.next_value)]
    parts[0] = "class TabularAgent:\n    # self.Q = np.zeros((1008, 3))\n" + parts[0]
    src = "\n\n".join(p.rstrip() for p in parts)
    return "\n".join(l for l in src.split("\n") if '"""' not in l and not l.strip().startswith("(ties"))


def training_loop_source():
    src = inspect.getsource(train.train)
    start = src.index("    for ep in range(episodes):")
    end = src.index("        if (ep + 1) % eval_every == 0:")
    return textwrap.dedent(src[start:end]).rstrip() + "\n    # every 2,000 episodes: greedy evaluation on 500 fixed validation nights"


def load_logs(algo):
    return [np.load(p, allow_pickle=True) for p in sorted(glob.glob(os.path.join(RES, f"{algo}_seed*_log.npz")))]


def fig_learning_curves(path):
    env = EVChargingEnv()
    val = [run_episode(env, pol, seed=train.VAL_SEED0 + i) for pol in (time_of_use, "oracle") for i in range(500)]
    ref = {"Time-of-use rule": val[:500], "Oracle (DP)": val[500:]}
    panels = [(1, "avg return per night", lambda r: r["return"], (-25, 0)),
              (3, "miss rate (%)", lambda r: r["missed"] * 100, (0, 40)),
              (2, "avg cost per night ($)", lambda r: r["cost"], (1.0, 4.5))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, (col, ylabel, key, ylim) in zip(axes, panels):
        for algo in ("qlearning", "sarsa"):
            ev = np.stack([lg["evals"] for lg in load_logs(algo)])        # seeds x evals x 4
            x, y = ev[0, :, 0] / 1000, ev[:, :, col] * (100 if col == 3 else 1)
            ax.plot(x, y.mean(0), color=COLORS[algo], lw=1.8, label=f"{NAMES[algo]} (mean of {len(y)} seeds)")
            ax.fill_between(x, y.min(0), y.max(0), color=COLORS[algo], alpha=0.2, lw=0)
        for (name, rs), ls in zip(ref.items(), ("--", ":")):
            ax.axhline(np.mean([key(r) for r in rs]), color="#555555", ls=ls, lw=1.1, label=name)
        ax.axvline(80, color="#BBBBBB", lw=0.8)
        ax.text(81, ylim[1] - 0.04 * (ylim[1] - ylim[0]), "ε reaches 0.05", fontsize=7, color="#888888", va="top")
        ax.set_ylim(*ylim); ax.set_xlabel("training episodes (thousands)", fontsize=9)
        ax.set_title(ylabel, fontsize=10); ax.tick_params(labelsize=8); ax.grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_baselines(path):
    rows = list(csv.DictReader(open(os.path.join(RES, "eval_results.csv"))))
    names = [r["policy"].replace(" (3 seeds)", "") for r in rows]
    colors = ["#BBBBBB", "#8C8C8C", "#6E6E6E", COLORS["qlearning"], COLORS["sarsa"], "#333333"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for ax, key, err, title, fmt in ((axes[0], "cost", "cost_std", "avg cost per night ($)", "${:.2f}"),
                                     (axes[1], "miss_rate", "miss_std", "miss rate (% of nights below 80%)", "{:.1f}%")):
        v = [float(r[key]) for r in rows]
        e = [float(r[err]) for r in rows]
        bars = ax.bar(range(len(rows)), v, yerr=[x if x > 0 else np.nan for x in e],
                      color=colors, capsize=3, width=0.65)
        for b, x, ex in zip(bars, v, e):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ex + max(v) * 0.02, fmt.format(x),
                    ha="center", fontsize=8)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, max(v) * 1.18)
        ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", alpha=0.25)
    axes[1].axhline(5, color="#C8463C", ls="--", lw=1)
    axes[1].text(5.4, 5.3, "5% goal", fontsize=7.5, color="#C8463C", ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    code_png(agent_source(), os.path.join(OUT, "fig1_agent_code.png"))
    code_png(training_loop_source(), os.path.join(OUT, "fig2_training_loop.png"))
    fig_learning_curves(os.path.join(OUT, "fig3_learning_curves.png"))
    fig_baselines(os.path.join(OUT, "fig4_baseline_comparison.png"))
    print("wrote figures to", OUT)
