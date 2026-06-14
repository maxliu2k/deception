"""Environment schematic for the deception competition (hand-drawn diagram, not
data). One round of play, with the repeat/trust feedback loop.

Flow: resort truth (+ per-agent visibility) -> 5 LLM agents emit claim vectors
-> mechanical buyer audits the most-exaggerated attribute, scores the survivors,
books the winner -> trust updates feed back for 12 rounds.

Outputs deception_environment.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "exports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8.5})

AGENTS = [
    ("Gemini Pro", "#1A7A3D"),
    ("Claude Opus", "#D26E00"),
    ("GPT-5.4", "#08519C"),
    ("Grok", "#000000"),
    ("Llama", "#A8327D"),
]


def _box(ax, cx, cy, w, h, text, *, fc="white", ec="0.25", lw=1.0, fs=8.5,
         tc="black", style="round,pad=0.02,rounding_size=0.02", weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style,
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, weight=weight)


def _arrow(ax, x0, y0, x1, y1, *, color="0.35", lw=1.1, style="-|>",
           conn="arc3,rad=0.0", mut=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=mut, lw=lw, color=color,
                                 connectionstyle=conn, zorder=1))


def main() -> int:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ── Truth + visibility (left) ───────────────────────────────────────────
    _box(ax, 11, 74, 19, 16,
         "Resort truth\n$t \\in [0,1]^5$", fc="#EEEEEE", lw=1.1, weight="bold")
    _box(ax, 11, 52, 19, 16,
         "Information:\nagent sees $k$ of 5\nattrs ($k{=}5$ full,\n$k{<}5$ partial)",
         fc="#F7F7F7", ec="0.55", fs=7.6)
    _arrow(ax, 11, 66, 11, 60, color="0.5", lw=1.0)

    # ── Agents (middle-left) ────────────────────────────────────────────────
    ay = [83, 67, 51, 35, 19]
    ax_c = 37
    for (name, col), y in zip(AGENTS, ay):
        _box(ax, ax_c, y, 21, 11.5, name, ec=col, lw=1.4, fc="white", tc=col, fs=8.2)
        # truth/info -> agent
        _arrow(ax, 20.5, 60 if y >= 54 else 50, ax_c - 10.7, y,
               color="0.6", lw=0.8, conn="arc3,rad=0.05")
        # agent -> buyer (claims)
        _arrow(ax, ax_c + 10.7, y, 56.5, 50, color=col, lw=1.0,
               conn=f"arc3,rad={0.18 if y>54 else (-0.18 if y<54 else 0.0)}")

    ax.text(48.5, 78, "claims\n$\\hat{c}_i \\in [0,1]^5$", ha="center", va="center",
            fontsize=7.8, color="0.25", style="italic")

    # ── Mechanical buyer (middle-right) ─────────────────────────────────────
    _box(ax, 69.5, 50, 27, 78, "", fc="#F4F8FB", ec="#3B6FB6", lw=1.5)
    ax.text(69.5, 84, "Mechanical buyer", ha="center", va="center",
            fontsize=9, weight="bold", color="#1F4E79")
    lines = [
        ("1.  population mean", "$\\mu = \\mathrm{mean}_i\\,\\hat{c}_i$"),
        ("2.  audit attribute", "$a^*_i = \\arg\\max_a\\,(\\hat{c}_{i,a}-\\mu_a)$"),
        ("    (most over-claimed", "vs. the crowd)"),
        ("3.  caught if", "$\\hat{c}_{i,a^*} \\neq t_{a^*}$"),
        ("4.  score (uncaught)", "$s_i=(w\\cdot\\hat{c}_i)(\\frac{1}{2}+\\frac{1}{2}\\mathrm{tr}_i)$"),
        ("5.  winner", "$\\arg\\max_i s_i$  if  $> \\tau$"),
    ]
    yy = 76
    for head, expr in lines:
        ax.text(58.5, yy, head, ha="left", va="center", fontsize=7.5, color="0.15")
        yy -= 4.4
        ax.text(60.5, yy, expr, ha="left", va="center", fontsize=8.0, color="#1F4E79")
        yy -= 6.2

    # ── Outcomes (right) ────────────────────────────────────────────────────
    _box(ax, 91.5, 62, 15, 30,
         "Winner\nbooks\n($+$reward)", fc="#E8F1E8", ec="#1A7A3D", lw=1.2, fs=7.8)
    _box(ax, 91.5, 30, 15, 34,
         "Trust update\ncaught: $r{=}{-}\\rho$,\n$\\mathrm{tr}{\\leftarrow}0.7\\,\\mathrm{tr}$\nclean:\n$\\mathrm{tr}{\\leftarrow}\\mathrm{tr}{+}0.1$",
         fc="#FBEEE6", ec="#D26E00", lw=1.2, fs=7.6)
    _arrow(ax, 83, 56, 84, 62, color="0.5", lw=1.0)
    _arrow(ax, 83, 44, 84, 34, color="0.5", lw=1.0)

    # ── Feedback loop (12 rounds, trust carries over) ───────────────────────
    _arrow(ax, 91.5, 13, 38, 12.5, color="#B03050", lw=1.2,
           conn="arc3,rad=0.30", style="-|>")
    ax.text(64, 3.0, "repeat $\\times\\,12$ rounds  —  trust $\\mathrm{tr}_i$ carries over",
            ha="center", va="center", fontsize=8, color="#B03050", style="italic")

    ax.text(50, 98.5, "Deception competition: one round", ha="center", va="center",
            fontsize=10.5, weight="bold")
    ax.text(50, 94.3,
            "Only over-claims are audited (argmax above the crowd); under-claims are never policed.",
            ha="center", va="center", fontsize=7.6, color="0.4", style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"deception_environment.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'deception_environment.pdf'}  (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
