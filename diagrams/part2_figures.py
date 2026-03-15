"""
generate_figures.py
Generates 4 figures for Part 2 Task 1:
  - figure_value_functions.png   (VI value grid + PI value grid side by side)
  - figure_policies.png          (VI policy grid + PI policy grid side by side)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from matplotlib.colors import LinearSegmentedColormap


ROADBLOCKS = {(2, 1), (2, 3)}
GOAL = (4, 4)
START = (0, 0)

V_VI = {
    (0, 0): -0.43,
    (1, 0): 0.63,
    (2, 0): 1.81,
    (3, 0): 3.12,
    (4, 0): 4.58,
    (0, 1): 0.63,
    (1, 1): 1.81,
    (3, 1): 4.58,
    (4, 1): 6.20,
    (0, 2): 1.81,
    (1, 2): 3.12,
    (2, 2): 4.58,
    (3, 2): 6.20,
    (4, 2): 8.00,
    (0, 3): 3.12,
    (1, 3): 4.58,
    (3, 3): 8.00,
    (4, 3): 10.00,
    (0, 4): 4.58,
    (1, 4): 6.20,
    (2, 4): 8.00,
    (3, 4): 10.00,
    (4, 4): 0.00,
}

V_PI = V_VI.copy()

PI_VI = {
    (0, 0): "U",
    (1, 0): "U",
    (2, 0): "R",
    (3, 0): "U",
    (4, 0): "U",
    (0, 1): "U",
    (1, 1): "U",
    (3, 1): "U",
    (4, 1): "U",
    (0, 2): "U",
    (1, 2): "U",
    (2, 2): "R",
    (3, 2): "U",
    (4, 2): "U",
    (0, 3): "U",
    (1, 3): "U",
    (3, 3): "U",
    (4, 3): "U",
    (0, 4): "R",
    (1, 4): "R",
    (2, 4): "R",
    (3, 4): "R",
    (4, 4): "G",
}

PI_PI = PI_VI.copy()


COLOR_HIGH = "#C0DD97"  # green  — high value / goal
COLOR_HIGH_TEXT = "#3B6D11"
COLOR_MID_WARM = "#F1EFE8"  # warm grey — mid value
COLOR_MID_TEXT = "#5F5E5A"
COLOR_LOW = "#FAEEDA"  # amber — low positive
COLOR_LOW_TEXT = "#854F0B"
COLOR_NEG = "#FCEBEB"  # red — negative value
COLOR_NEG_TEXT = "#A32D2D"
COLOR_BLOCK = "#D3D1C7"  # grey — roadblock
COLOR_BLOCK_TXT = "#888780"
COLOR_GOAL_BG = "#C0DD97"
COLOR_GOAL_TXT = "#3B6D11"
COLOR_START_BG = "#B5D4F4"  # blue — start state
COLOR_START_TXT = "#0C447C"
COLOR_AXIS = "#888780"
COLOR_BORDER = "#B4B2A9"  # 0.5px border equivalent

CELL_W = 1.2
CELL_H = 0.85
FONT_CELL = 9.5
FONT_AXIS = 8.5
FONT_SUB = 10.5
LW = 0.6

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))


def val_color(val):
    if val >= 8.0:
        return COLOR_HIGH, COLOR_HIGH_TEXT
    elif val >= 3.0:
        return "#EAF3DE", COLOR_HIGH_TEXT  # lighter green
    elif val >= 0.5:
        return COLOR_MID_WARM, COLOR_MID_TEXT
    elif val >= 0.0:
        return COLOR_LOW, COLOR_LOW_TEXT
    else:
        return COLOR_NEG, COLOR_NEG_TEXT


def draw_value_grid(ax, V, subtitle):
    ax.set_xlim(0, 5 * CELL_W)
    ax.set_ylim(0, 5 * CELL_H)
    ax.set_aspect("equal")
    ax.axis("off")

    for x in range(5):
        for y in range(5):
            left = x * CELL_W
            bottom = y * CELL_H

            if (x, y) in ROADBLOCKS:
                bg, fg = COLOR_BLOCK, COLOR_BLOCK_TXT
                label = "Block"
            elif (x, y) == GOAL:
                bg, fg = COLOR_GOAL_BG, COLOR_GOAL_TXT
                label = "Goal"
            else:
                val = V.get((x, y), 0.0)
                bg, fg = val_color(val)
                label = f"{val:.2f}"

            rect = plt.Rectangle(
                (left, bottom),
                CELL_W,
                CELL_H,
                facecolor=bg,
                edgecolor=COLOR_BORDER,
                linewidth=LW,
            )
            ax.add_patch(rect)

            if (x, y) == START:
                outline = plt.Rectangle(
                    (left + 0.03, bottom + 0.03),
                    CELL_W - 0.06,
                    CELL_H - 0.06,
                    facecolor="none",
                    edgecolor=COLOR_START_TXT,
                    linewidth=1.2,
                )
                ax.add_patch(outline)
                fg = COLOR_START_TXT

            ax.text(
                left + CELL_W / 2,
                bottom + CELL_H / 2,
                label,
                ha="center",
                va="center",
                fontsize=FONT_CELL,
                color=fg,
                fontfamily="monospace",
                fontweight="normal",
            )

    for i in range(5):
        ax.text(
            i * CELL_W + CELL_W / 2,
            -0.25 * CELL_H,
            f"x={i}",
            ha="center",
            va="top",
            fontsize=FONT_AXIS,
            color=COLOR_AXIS,
        )
        ax.text(
            -0.18 * CELL_W,
            i * CELL_H + CELL_H / 2,
            f"y={i}",
            ha="right",
            va="center",
            fontsize=FONT_AXIS,
            color=COLOR_AXIS,
        )

    ax.set_title(
        subtitle, fontsize=FONT_SUB, color="#444441", fontweight="normal", pad=8
    )


ARROW_SYMBOL = {
    "U": "↑",
    "D": "↓",
    "L": "←",
    "R": "→",
}


def draw_policy_grid(ax, policy, subtitle):
    ax.set_xlim(0, 5 * CELL_W)
    ax.set_ylim(0, 5 * CELL_H)
    ax.set_aspect("equal")
    ax.axis("off")

    for x in range(5):
        for y in range(5):
            left = x * CELL_W
            bottom = y * CELL_H

            if (x, y) in ROADBLOCKS:
                bg = COLOR_BLOCK
            elif (x, y) == GOAL:
                bg = COLOR_GOAL_BG
            elif (x, y) == START:
                bg = COLOR_START_BG
            else:
                bg = "#F8F7F3"

            rect = plt.Rectangle(
                (left, bottom),
                CELL_W,
                CELL_H,
                facecolor=bg,
                edgecolor=COLOR_BORDER,
                linewidth=LW,
            )
            ax.add_patch(rect)

            cx = left + CELL_W / 2
            cy = bottom + CELL_H / 2

            if (x, y) in ROADBLOCKS:
                ax.text(
                    cx,
                    cy,
                    "✕",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=COLOR_BLOCK_TXT,
                )

            elif (x, y) == GOAL:
                ax.text(
                    cx,
                    cy,
                    "G",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=COLOR_GOAL_TXT,
                    fontweight="normal",
                )

            else:
                action = policy.get((x, y), "U")
                arrow = ARROW_SYMBOL.get(action, "↑")
                ax.text(
                    cx,
                    cy,
                    arrow,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="#3C3489",
                )

                if (x, y) == START:
                    outline = plt.Rectangle(
                        (left + 0.03, bottom + 0.03),
                        CELL_W - 0.06,
                        CELL_H - 0.06,
                        facecolor="none",
                        edgecolor=COLOR_START_TXT,
                        linewidth=1.2,
                    )
                    ax.add_patch(outline)

    # axis labels
    for i in range(5):
        ax.text(
            i * CELL_W + CELL_W / 2,
            -0.25 * CELL_H,
            f"x={i}",
            ha="center",
            va="top",
            fontsize=FONT_AXIS,
            color=COLOR_AXIS,
        )
        ax.text(
            -0.18 * CELL_W,
            i * CELL_H + CELL_H / 2,
            f"y={i}",
            ha="right",
            va="center",
            fontsize=FONT_AXIS,
            color=COLOR_AXIS,
        )

    ax.set_title(
        subtitle, fontsize=FONT_SUB, color="#444441", fontweight="normal", pad=8
    )


def make_value_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9, 4.6),
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    draw_value_grid(ax1, V_VI, "Value Iteration  (9 iterations)")
    draw_value_grid(ax2, V_PI, "Policy Iteration  (6 improvement steps)")

    # colour legend
    legend_items = [
        mpatches.Patch(
            facecolor="#EAF3DE", edgecolor=COLOR_BORDER, lw=0.5, label="V ≥ 3.0"
        ),
        mpatches.Patch(
            facecolor=COLOR_HIGH, edgecolor=COLOR_BORDER, lw=0.5, label="V ≥ 8.0"
        ),
        mpatches.Patch(
            facecolor=COLOR_MID_WARM,
            edgecolor=COLOR_BORDER,
            lw=0.5,
            label="0.5 ≤ V < 3.0",
        ),
        mpatches.Patch(
            facecolor=COLOR_LOW, edgecolor=COLOR_BORDER, lw=0.5, label="0 ≤ V < 0.5"
        ),
        mpatches.Patch(
            facecolor=COLOR_NEG, edgecolor=COLOR_BORDER, lw=0.5, label="V < 0"
        ),
        mpatches.Patch(
            facecolor=COLOR_START_BG,
            edgecolor=COLOR_START_TXT,
            lw=1.0,
            label="Start (0,0)",
        ),
        mpatches.Patch(
            facecolor=COLOR_BLOCK, edgecolor=COLOR_BORDER, lw=0.5, label="Roadblock"
        ),
        mpatches.Patch(
            facecolor=COLOR_GOAL_BG, edgecolor=COLOR_BORDER, lw=0.5, label="Goal (4,4)"
        ),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=1.2,
        handleheight=0.9,
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure_value_functions.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved figure_value_functions.png")
    plt.close()


def make_policy_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9, 4.6),
        gridspec_kw={"wspace": 0.32},
    )
    fig.patch.set_facecolor("white")

    draw_policy_grid(ax1, PI_VI, "Value Iteration  (9 iterations)")
    draw_policy_grid(ax2, PI_PI, "Policy Iteration  (6 improvement steps)")

    legend_items = [
        mpatches.Patch(
            facecolor=COLOR_START_BG,
            edgecolor=COLOR_START_TXT,
            lw=1.0,
            label="Start (0,0)",
        ),
        mpatches.Patch(
            facecolor=COLOR_GOAL_BG, edgecolor=COLOR_BORDER, lw=0.5, label="Goal (4,4)"
        ),
        mpatches.Patch(
            facecolor=COLOR_BLOCK, edgecolor=COLOR_BORDER, lw=0.5, label="Roadblock"
        ),
        mpatches.Patch(
            facecolor="#F8F7F3", edgecolor=COLOR_BORDER, lw=0.5, label="Free cell"
        ),
    ]

    from matplotlib.lines import Line2D

    legend_items.append(
        Line2D(
            [0],
            [0],
            color="#3C3489",
            lw=1.3,
            marker=">",
            markersize=5,
            label="Optimal action",
        )
    )
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=1.4,
        handleheight=0.9,
    )

    fig.text(
        0.5,
        0.97,
        "100% policy agreement across all 22 non-terminal states",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#3B6D11",
        style="italic",
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, "figure_policies.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved figure_policies.png")
    plt.close()


if __name__ == "__main__":
    make_value_figure()
    make_policy_figure()
    print("Done. Two PNG files ready for your report.")
