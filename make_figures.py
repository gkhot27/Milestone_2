"""Generates the report figures into figures/.

fig1_episodes.png     two example nights, all three baselines
fig2_step_code.png    the step() source of EVChargingEnv (read live from ev_env.py)
fig3_test_output.png  test_output.txt + baseline_results.txt

Run test_env.py and baselines.py first so the .txt files are up to date:
    python3 test_env.py > test_output.txt
    python3 baselines.py > baseline_results.txt
"""
import inspect
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pygments import lex
from pygments.lexers import PythonLexer, TextLexer
from pygments.styles import get_style_by_name

from ev_env import EVChargingEnv
from baselines import charge_immediately, time_of_use, oracle_plan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

PRICE_COLORS = ["#4C9F70", "#E0A030", "#C8463C"]          # low, medium, high
POLICIES = [("Charge immediately", charge_immediately, "#5B6C8F", "-"),
            ("Time-of-use rule", time_of_use, "#8E5BA8", "--"),
            ("Oracle (DP)", "oracle", "#222222", ":")]
# Seeds chosen to show (1) a normal night and (2) a surprise early departure.
EPISODES = [(37, "Normal night"), (79, "Early departure")]


def trace(env, policy, seed):
    """Run one episode; return clock hours, battery % after each hour, and the final info."""
    obs, info = env.reset(seed=seed)
    plan = oracle_plan(env)[0] if policy == "oracle" else None
    clocks, batt = [env.clock], [env.battery * 100]
    done, t = False, 0
    while not done:
        a = plan[t] if plan is not None else policy(obs, info, env)
        obs, _, done, _, info = env.step(a)
        clocks.append(env.clock)
        batt.append(env.battery * 100)
        t += 1
    return np.array(clocks), np.array(batt), info


def fig_episodes(path):
    env = EVChargingEnv()
    fig, axes = plt.subplots(2, len(EPISODES), figsize=(11, 6.6), sharex="col",
                             gridspec_kw={"height_ratios": [1, 2]})
    for col, (seed, title) in enumerate(EPISODES):
        env.reset(seed=seed)
        plug, planned, actual = env.plug_in, env.planned_departure, env.actual_departure
        b0 = env.battery * 100
        hours = np.arange(plug, planned)

        ax = axes[0, col]
        ax.bar(hours + 0.5, env.prices[env.price_path], width=0.9,
               color=[PRICE_COLORS[p] for p in env.price_path])
        ax.set_ylabel("price ($/kWh)", fontsize=8)
        ax.set_ylim(0, 0.36)
        ax.tick_params(labelsize=7)
        fmt = lambda h: f"{(h % 24) % 12 or 12}{'am' if h % 24 < 12 else 'pm'}"
        if actual < planned:
            sub = (f"{fmt(plug)} at {b0:.0f}%; planned {fmt(planned)}, "
                   f"leaves {fmt(actual)}")
        else:
            sub = f"plug in {fmt(plug)} at {b0:.0f}%, leave {fmt(planned)}"
        ax.set_title(f"{title} ({sub})", fontsize=9)

        ax = axes[1, col]
        for name, pol, color, ls in POLICIES:
            clk, b, info = trace(env, pol, seed)
            lbl = f"{name}: ${info['total_cost']:.2f}"
            if info["shortfall"] > 0.5:
                lbl += f", MISSED by {info['shortfall']:.0f}%"
            ax.step(clk, b, where="post", color=color, ls=ls, lw=1.4, label=lbl)
            ax.plot(clk[-1], b[-1], "o", color=color, ms=3.5)
        ax.axhline(80, color="gray", lw=0.8)
        ax.text(plug + 0.3, 81, "target 80%", fontsize=6.5, color="gray")
        ax.axvline(planned, color="gray", ls="--", lw=0.8)
        ax.text(planned - 0.15, 2, "planned\ndeparture", fontsize=6.5, color="gray", ha="right")
        if actual < planned:
            ax.axvline(actual, color="#C8463C", lw=1.0)
            ax.text(actual - 0.15, 2, "driver leaves\nearly", fontsize=6.5,
                    color="#C8463C", ha="right")
        ax.set_ylim(0, 103)
        ax.set_ylabel("battery (%)", fontsize=8)
        ticks = np.arange(plug, planned + 1, 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t % 24}:00" for t in ticks], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)

    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(PRICE_COLORS, ["low", "medium", "high"])],
               loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_text(text, path, lexer, line_numbers=False, width_in=9.0):
    """Render code/terminal text to a PNG (used for the code + output screenshots)."""
    style = get_style_by_name("default")
    lines = text.rstrip("\n").split("\n")
    lh = 0.17                                   # inches per line
    fig = plt.figure(figsize=(width_in, lh * len(lines) + 0.3))
    fig.patch.set_facecolor("#F0F0F0")
    renderer = fig.canvas.get_renderer()
    x0 = 0.45 if line_numbers else 0.15
    y = fig.get_figheight() - 0.2

    def put(x, s, **kw):
        return fig.text(x / width_in, y / fig.get_figheight(), s, family="monospace",
                        fontsize=8, va="top", ha="left", **kw)

    # monospace font: measure one character width and place tokens by column
    probe = put(0, "M" * 50)
    char_w = probe.get_window_extent(renderer).width / fig.dpi / 50
    probe.remove()

    for i, line in enumerate(lines, 1):
        if line_numbers:
            put(0.05, f"{i:>2}", color="#888888")
        col = 0
        for ttype, value in lex(line, lexer):
            value = value.rstrip("\n")
            if not value:
                continue
            if value.strip():
                st = style.style_for_token(ttype)
                put(x0 + col * char_w, value,
                    color=f"#{st['color']}" if st["color"] else "#000000",
                    fontweight="bold" if st["bold"] else "normal",
                    fontstyle="italic" if st["italic"] else "normal")
            col += len(value)
        y -= lh
    if line_numbers:
        fig.add_artist(plt.Line2D([0.38 / width_in] * 2, [0, 1], color="#CCCCCC", lw=0.8))
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def step_source():
    """step() from ev_env.py, with the history/render bookkeeping dropped and the
    info call shortened so it fits the figure (same trimming as the report)."""
    src = textwrap.dedent(inspect.getsource(EVChargingEnv.step))
    keep = []
    for line in src.split("\n"):
        s = line.strip()
        if s.startswith(("self.history.append", "if self.render_mode", "print(self.render")):
            continue
        if s.startswith("self._info(energy"):
            continue
        if s.startswith("return (self._obs()"):
            line = line.replace("False,", "False, self._info(...))")
        keep.append(line)
    out = "\n".join(keep).rstrip()
    # put a blank line before the return, matching the report layout
    return out.replace("\n    return (", "\n\n    return (").replace("\n\n\n", "\n\n")


def fig2_step_code(path):
    """Syntax-highlighted step() via Pygments' image formatter (needs Pillow)."""
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    kw = dict(line_numbers=True, font_size=22, line_number_bg="#E8EAD8",
              line_number_fg="#888888", image_pad=18, style="default")
    for font in ("Menlo", "DejaVu Sans Mono", "Consolas", "Courier New"):
        try:
            png = highlight(step_source(), PythonLexer(), ImageFormatter(font_name=font, **kw))
            break
        except Exception:
            continue
    with open(path, "wb") as f:
        f.write(png)


def fig_test_output(path):
    read = lambda f: open(os.path.join(HERE, f)).read().rstrip()
    text = (read("test_output.txt") + "\n\n$ python3 baselines.py   (5,000 episodes each)\n"
            + read("baseline_results.txt"))
    render_text(text, path, TextLexer(), width_in=9.0)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_episodes(os.path.join(OUT, "fig1_episodes.png"))
    fig2_step_code(os.path.join(OUT, "fig2_step_code.png"))
    fig_test_output(os.path.join(OUT, "fig3_test_output.png"))
    print("wrote figures to", OUT)
