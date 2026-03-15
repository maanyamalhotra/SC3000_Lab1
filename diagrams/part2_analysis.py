"""
Convergence Analysis for Part 2 Report

Tracks policy agreement with the optimal (VI) policy
at regular checkpoints during MC and Q-Learning training
and produces convergence plot.

"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

from part2 import (
    GridWorld,
    value_iteration,
    ACTIONS,
    GAMMA,
    ALPHA,
    EPSILON,
    GOAL,
    ROADBLOCKS,
)


# helpers
def extract_greedy_policy(Q, states):
    """Extract greedy policy from Q-table."""
    pi = {}
    for s in states:
        if s == GOAL:
            pi[s] = "Goal"
            continue
        q_vals = [Q[(s, a)] for a in range(len(ACTIONS))]
        pi[s] = ACTIONS[int(np.argmax(q_vals))]
    return pi


def policy_agreement(pi, pi_optimal, states):
    """
    Return % of non-goal, non-roadblock states where
    pi matches pi_optimal.
    """
    comparable = [s for s in states if s != GOAL and s not in ROADBLOCKS]
    matches = sum(1 for s in comparable if pi.get(s) == pi_optimal.get(s))
    return 100.0 * matches / len(comparable)


def mc_with_checkpoints(
    env,
    pi_optimal,
    checkpoint_every=10,
    num_episodes=50000,
    gamma=GAMMA,
    epsilon=EPSILON,
    seed=42,
):
    """
    Runs MC control and records policy agreement with pi_optimal
    every `checkpoint_every` episodes.

    Returns:
        episodes  list of episode numbers at each checkpoint
        agreement list of % agreement at each checkpoint
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    Q = defaultdict(float)
    returns_count = defaultdict(int)

    def epsilon_greedy(state):
        if rng.random() < epsilon:
            return rng.randrange(len(ACTIONS))
        q_vals = [Q[(state, a)] for a in range(len(ACTIONS))]
        max_q = max(q_vals)
        best = [a for a, q in enumerate(q_vals) if q == max_q]
        return rng.choice(best)

    episodes_log, agreement_log = [], []

    for ep in range(1, num_episodes + 1):
        episode = []
        state = env.start
        for _ in range(500):
            a_idx = epsilon_greedy(state)
            next_state, reward, done = env.step(state, ACTIONS[a_idx])
            episode.append((state, a_idx, reward))
            state = next_state
            if done:
                break

        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            sa = (s_t, a_t)
            if sa not in visited:
                visited.add(sa)
                returns_count[sa] += 1
                Q[sa] += (G - Q[sa]) / returns_count[sa]

        if ep % checkpoint_every == 0:
            pi_current = extract_greedy_policy(Q, env.states)
            agreement_log.append(policy_agreement(pi_current, pi_optimal, env.states))
            episodes_log.append(ep)

    return episodes_log, agreement_log


def ql_with_checkpoints(
    env,
    pi_optimal,
    checkpoint_every=10,
    num_episodes=50000,
    gamma=GAMMA,
    alpha=ALPHA,
    epsilon=EPSILON,
    seed=42,
):
    """
    Runs Q-Learning and records policy agreement with pi_optimal
    every `checkpoint_every` episodes.

    Returns:
        episodes  – list of episode numbers at each checkpoint
        agreement – list of % agreement at each checkpoint
    """
    rng = random.Random(seed)
    np.random.seed(seed + 1)

    Q = defaultdict(float)

    episodes_log, agreement_log = [], []

    for ep in range(1, num_episodes + 1):
        state = env.start
        for _ in range(500):
            # epsilon-greedy action selection
            if rng.random() < epsilon:
                a_idx = rng.randrange(len(ACTIONS))
            else:
                q_vals = [Q[(state, a)] for a in range(len(ACTIONS))]
                max_q = max(q_vals)
                best = [a for a, q in enumerate(q_vals) if q == max_q]
                a_idx = rng.choice(best)

            next_state, reward, done = env.step(state, ACTIONS[a_idx])

            # TD update
            max_q_next = max(Q[(next_state, a)] for a in range(len(ACTIONS)))
            td_target = reward + gamma * max_q_next
            Q[(state, a_idx)] += alpha * (td_target - Q[(state, a_idx)])

            state = next_state
            if done:
                break

        if ep % checkpoint_every == 0:
            pi_current = extract_greedy_policy(Q, env.states)
            agreement_log.append(policy_agreement(pi_current, pi_optimal, env.states))
            episodes_log.append(ep)

    return episodes_log, agreement_log


def plot_convergence(mc_eps, mc_agree, ql_eps, ql_agree):
    """
    Produces a convergence plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    MC_COLOR = "#8054E0"
    QL_COLOR = "#4A90D9"
    OPT_COLOR = "#2ECC71"

    ax.plot(
        mc_eps,
        mc_agree,
        color=MC_COLOR,
        linewidth=2,
        label="Monte Carlo (ε-greedy, first-visit)",
        zorder=3,
    )
    ax.plot(
        ql_eps,
        ql_agree,
        color=QL_COLOR,
        linewidth=2,
        label="Q-Learning (ε-greedy, off-policy TD)",
        zorder=3,
    )
    ax.axhline(
        y=100,
        color=OPT_COLOR,
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label="Optimal (VI/PI baseline)",
        zorder=2,
    )

    for eps_log, agree_log, color, name in [
        (mc_eps, mc_agree, MC_COLOR, "MC"),
        (ql_eps, ql_agree, QL_COLOR, "QL"),
    ]:
        threshold = 90.0
        for i, (ep, ag) in enumerate(zip(eps_log, agree_log)):
            if ag >= threshold:
                ax.axvline(x=ep, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
                ax.annotate(
                    f"{name} ≥90%\n@ ep {ep:,}",
                    xy=(ep, threshold),
                    xytext=(ep + 800, threshold - 12),
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.9,
                    ),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                )
                break

    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Policy Agreement with Optimal (%)", fontsize=11)
    ax.set_title(
        "Convergence of MC and Q-Learning to Optimal Policy\n"
        r"5×5 GridWorld  |  $\gamma=0.9$,  $\varepsilon=0.1$,  $\alpha=0.1$",
        fontsize=12,
        pad=10,
    )

    ax.set_xlim(0, max(mc_eps[-1], ql_eps[-1]))
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    final_mc = mc_agree[-1]
    final_ql = ql_agree[-1]
    ax.fill_between(
        mc_eps, mc_agree, ql_agree, alpha=0.07, color="grey", label="_nolegend_"
    )

    plt.tight_layout()
    output_path = "convergence_plot.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close()

    return final_mc, final_ql


if __name__ == "__main__":
    print("setting up environment...")
    env = GridWorld()
    V_opt, pi_opt = value_iteration(env)
    mc_eps, mc_agree = mc_with_checkpoints(env, pi_opt, checkpoint_every=10)
    ql_eps, ql_agree = ql_with_checkpoints(env, pi_opt, checkpoint_every=10)
    final_mc, final_ql = plot_convergence(mc_eps, mc_agree, ql_eps, ql_agree)

    print("\nConvergence Summary")
    print(f"  Monte Carlo - final agreement: {final_mc:.1f}%")
    print(f"  Q-Learning - final agreement: {final_ql:.1f}%")

    # find crossover point
    for i, (ep, mc_a, ql_a) in enumerate(zip(mc_eps, mc_agree, ql_agree)):
        if ql_a > mc_a and i > 0:
            print(f"Q-Learning overtakes MC around episode {ep:,}")
            break
