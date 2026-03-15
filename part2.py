"""
SC3000 Lab Assignment 1 — Part 2: Grid World MDP & Reinforcement Learning

Environment:
    5x5 grid, start=(0,0), goal=(4,4)
    Roadblocks at (2,1) and (2,3)
    Actions: Up, Down, Left, Right
    Reward: -1 per step, +10 on reaching goal
    Stochastic transitions: 0.8 intended, 0.1 each perpendicular

Algorithms:
    1a. Value Iteration  (deterministic transitions)
    1b. Policy Iteration (deterministic transitions)
    2.  Monte Carlo Control (stochastic, ε-greedy, first-visit)
    3.  Q-Learning (stochastic, ε-greedy, TD)
"""

import random
import time
import numpy as np
from collections import defaultdict

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
ROADBLOCKS = {(2, 1), (2, 3)}

ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# direction vectors (dx, dy)
ACTION_DELTAS = {
    "Up": (0, 1),
    "Down": (0, -1),
    "Left": (-1, 0),
    "Right": (1, 0),
}

# perpendicular actions for stochastic transitions
PERPENDICULAR = {
    "Up": ("Left", "Right"),
    "Down": ("Left", "Right"),
    "Left": ("Down", "Up"),
    "Right": ("Down", "Up"),
}

# parameters
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1

# arrow symbols for policy display
ARROW = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→"}


# gridworld Environment
class GridWorld:
    """
    5x5 Grid World MDP environment.

    States are (x, y) with x, y ∈ {0, 1, 2, 3, 4}.
    Origin (0,0) is bottom-left; y increases upward.
    """

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start = START
        self.goal = GOAL
        self.roadblocks = ROADBLOCKS

        # all valid states
        self.states = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in ROADBLOCKS
        ]

    # -- helpers --

    def _is_valid(self, state):
        """Check if a state is within bounds and not a roadblock."""
        x, y = state
        return (
            0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and state not in self.roadblocks
        )

    def _attempt_move(self, state, action):
        """Return the resulting state after attempting *action* from *state*.
        If the move is invalid, the agent stays in place."""
        dx, dy = ACTION_DELTAS[action]
        new_state = (state[0] + dx, state[1] + dy)
        return new_state if self._is_valid(new_state) else state

    # -- transition model --

    def get_transition_probs(self, state, action, deterministic=False):
        """
        Return a list of (probability, next_state) for taking *action* in *state*.

        If deterministic=True, the intended direction succeeds with prob 1.0.
        Otherwise, stochastic: 0.8 intended, 0.1 each perpendicular.
        """
        if state == self.goal:
            return [(1.0, self.goal)]  # terminal / absorbing

        if deterministic:
            return [(1.0, self._attempt_move(state, action))]

        perp_left, perp_right = PERPENDICULAR[action]
        outcomes = [
            (0.8, self._attempt_move(state, action)),
            (0.1, self._attempt_move(state, perp_left)),
            (0.1, self._attempt_move(state, perp_right)),
        ]
        # merge duplicate next-states (two moves that both bounce back)
        merged = defaultdict(float)
        for prob, s_next in outcomes:
            merged[s_next] += prob
        return [(p, s) for s, p in merged.items()]

    def step(self, state, action):
        """
        Sample one transition stochastically.
        Returns (next_state, reward, done).
        """
        if state == self.goal:
            return self.goal, 0.0, True

        transitions = self.get_transition_probs(state, action, deterministic=False)
        probs, next_states = zip(*transitions)
        idx = np.random.choice(len(probs), p=probs)
        s_next = next_states[idx]

        reward = 10.0 if s_next == self.goal else -1.0
        done = s_next == self.goal
        return s_next, reward, done

    def get_reward(self, state, action, next_state):
        """Return the reward for transitioning from state to next_state."""
        if state == self.goal:
            return 0.0
        return 10.0 if next_state == self.goal else -1.0


# 1a. val iteration  (deterministic transitions)
def value_iteration(env, gamma=GAMMA, theta=1e-6):
    """
    Compute optimal value function V* and policy π* using
    the Bellman optimality equation with deterministic transitions.

    V(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]

    Returns:
        V   – dict {state: value}
        pi  – dict {state: action}
    """
    V = {s: 0.0 for s in env.states}
    iteration = 0

    while True:
        delta = 0.0
        for s in env.states:
            if s == env.goal:
                continue
            v_old = V[s]
            action_values = []
            for a in ACTIONS:
                q = 0.0
                for prob, s_next in env.get_transition_probs(s, a, deterministic=True):
                    r = env.get_reward(s, a, s_next)
                    q += prob * (r + gamma * V[s_next])
                action_values.append(q)
            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        iteration += 1
        if delta < theta:
            break

    # extract policy pi
    pi = {}
    for s in env.states:
        if s == env.goal:
            pi[s] = "Goal"
            continue
        best_a, best_q = None, -float("inf")
        for a in ACTIONS:
            q = 0.0
            for prob, s_next in env.get_transition_probs(s, a, deterministic=True):
                r = env.get_reward(s, a, s_next)
                q += prob * (r + gamma * V[s_next])
            if q > best_q:
                best_a, best_q = a, q
        pi[s] = best_a

    print(f"  Value Iteration converged in {iteration} iterations.")
    return V, pi


# 1b. policy iteration  (deterministic transitions)


def policy_iteration(env, gamma=GAMMA, theta=1e-6):
    """
    Compute optimal value function V* and policy π* via
    alternating policy evaluation and policy improvement,
    using deterministic transitions.

    Returns:
        V   – dict {state: value}
        pi  – dict {state: action}
    """
    # arbitrary policy (up)
    pi = {s: "Up" for s in env.states}
    pi[env.goal] = "Goal"
    V = {s: 0.0 for s in env.states}
    iteration = 0

    while True:
        # ─-- policy eval --
        while True:
            delta = 0.0
            for s in env.states:
                if s == env.goal:
                    continue
                v_old = V[s]
                a = pi[s]
                q = 0.0
                for prob, s_next in env.get_transition_probs(s, a, deterministic=True):
                    r = env.get_reward(s, a, s_next)
                    q += prob * (r + gamma * V[s_next])
                V[s] = q
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break

        # -- policy improvement --
        stable = True
        for s in env.states:
            if s == env.goal:
                continue
            old_action = pi[s]
            best_a, best_q = None, -float("inf")
            for a in ACTIONS:
                q = 0.0
                for prob, s_next in env.get_transition_probs(s, a, deterministic=True):
                    r = env.get_reward(s, a, s_next)
                    q += prob * (r + gamma * V[s_next])
                if q > best_q:
                    best_a, best_q = a, q
            pi[s] = best_a
            if old_action != best_a:
                stable = False
        iteration += 1
        if stable:
            break

    print(f"  Policy Iteration converged in {iteration} improvement steps.")
    return V, pi


# 2. Monte Carlo control  (stochastic, first-visit, ε-greedy)


def monte_carlo_control(
    env, gamma=GAMMA, epsilon=EPSILON, num_episodes=50000, max_steps=500, seed=42
):
    """
    First-visit Monte Carlo control with ε-greedy exploration.

    Uses the stochastic environment (env.step) to generate episodes
    and learns Q(s,a) from sampled returns.

    Returns:
        Q           – dict {(state, action_idx): value}
        pi          – dict {state: action}
        num_episodes – number of episodes trained
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Q-values and visit counts
    Q = defaultdict(float)
    returns_count = defaultdict(int)

    # ε-greedy policy
    def epsilon_greedy(state):
        if rng.random() < epsilon:
            return rng.randrange(len(ACTIONS))
        q_vals = [Q[(state, a)] for a in range(len(ACTIONS))]
        max_q = max(q_vals)
        best = [a for a, q in enumerate(q_vals) if q == max_q]
        return rng.choice(best)

    # monkey-patch env's rng for reproducibility within episodes
    old_rng_state = np.random.get_state()
    np.random.seed(seed)

    for ep in range(num_episodes):
        # generate episode
        episode = []
        state = env.start
        for _ in range(max_steps):
            a_idx = epsilon_greedy(state)
            next_state, reward, done = env.step(state, ACTIONS[a_idx])
            episode.append((state, a_idx, reward))
            state = next_state
            if done:
                break

        # Compute discounted returns for every step (backward pass)
        returns = [0.0] * len(episode)
        G = 0.0
        for t in reversed(range(len(episode))):
            _, _, r_t = episode[t]
            G = gamma * G + r_t
            returns[t] = G

        # First-visit MC update: forward pass, update only the first occurrence of each (s,a)
        visited = set()
        for t in range(len(episode)):
            s_t, a_t, _ = episode[t]
            sa = (s_t, a_t)
            if sa not in visited:
                visited.add(sa)
                returns_count[sa] += 1
                # incremental mean update
                Q[sa] += (returns[t] - Q[sa]) / returns_count[sa]

    np.random.set_state(old_rng_state)  # restore

    # extract greedy policy
    pi = {}
    for s in env.states:
        if s == env.goal:
            pi[s] = "Goal"
            continue
        q_vals = [Q[(s, a)] for a in range(len(ACTIONS))]
        pi[s] = ACTIONS[int(np.argmax(q_vals))]

    print(f"  Monte Carlo Control trained for {num_episodes} episodes.")
    return Q, pi, num_episodes


# 3. Q-Learning  (stochastic, ε-greedy, off-policy TD)


def q_learning(
    env,
    gamma=GAMMA,
    alpha=ALPHA,
    epsilon=EPSILON,
    num_episodes=50000,
    max_steps=500,
    seed=42,
):
    """
    Tabular Q-Learning with ε-greedy exploration.

    Update: Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]

    Returns:
        Q           – dict {(state, action_idx): value}
        pi          – dict {state: action}
        num_episodes – number of episodes trained
    """
    rng = random.Random(seed)

    Q = defaultdict(float)

    old_rng_state = np.random.get_state()
    np.random.seed(seed + 1)  # different seed

    for ep in range(num_episodes):
        state = env.start
        for _ in range(max_steps):
            # ε-greedy action selection
            if rng.random() < epsilon:
                a_idx = rng.randrange(len(ACTIONS))
            else:
                q_vals = [Q[(state, a)] for a in range(len(ACTIONS))]
                max_q = max(q_vals)
                best = [a for a, q in enumerate(q_vals) if q == max_q]
                a_idx = rng.choice(best)

            next_state, reward, done = env.step(state, ACTIONS[a_idx])

            # Q-learning update (off-policy: use max over next actions)
            max_q_next = max(Q[(next_state, a)] for a in range(len(ACTIONS)))
            td_target = reward + gamma * max_q_next
            Q[(state, a_idx)] += alpha * (td_target - Q[(state, a_idx)])

            state = next_state
            if done:
                break

    np.random.set_state(old_rng_state)

    # etract greedy policy
    pi = {}
    for s in env.states:
        if s == env.goal:
            pi[s] = "Goal"
            continue
        q_vals = [Q[(s, a)] for a in range(len(ACTIONS))]
        pi[s] = ACTIONS[int(np.argmax(q_vals))]

    print(f"  Q-Learning trained for {num_episodes} episodes.")
    return Q, pi, num_episodes


# display helper funcs


def print_value_function(V, title="Value Function"):
    """Print V as a 5×5 grid (row 4 = top / y=4, row 0 = bottom / y=0)."""
    print(f"\n  {title}:")
    print("  " + "-" * 46)
    for y in reversed(range(GRID_SIZE)):
        row = []
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row.append("  XXXX ")
            elif s in V:
                row.append(f"{V[s]:7.2f}")
            else:
                row.append("  XXXX ")
        print(f"  y={y} | " + " | ".join(row) + " |")
    print("  " + "-" * 46)
    print("       " + "   ".join(f"x={x}" for x in range(GRID_SIZE)))


def print_policy(pi, title="Policy"):
    """Print policy as a 5×5 grid with arrows."""
    print(f"\n  {title}:")
    print("  " + "-" * 31)
    for y in reversed(range(GRID_SIZE)):
        row = []
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row.append(" X ")
            elif s == GOAL:
                row.append(" G ")
            elif s in pi:
                row.append(f" {ARROW.get(pi[s], '?')} ")
            else:
                row.append(" X ")
        print(f"  y={y} |" + "|".join(row) + "|")
    print("  " + "-" * 31)
    print("      " + "  ".join(f"x={x}" for x in range(GRID_SIZE)))


def q_to_v(Q, states):
    """Convert Q-table to value function by taking max over actions."""
    V = {}
    for s in states:
        V[s] = max(Q[(s, a)] for a in range(len(ACTIONS)))
    return V


def compare_policies(policies, names):
    """Print a comparison table showing where policies agree/disagree."""
    print("\n  Policy Comparison:")
    agree = 0
    disagree = 0
    for s in sorted(policies[0].keys()):
        if s == GOAL or s in ROADBLOCKS:
            continue
        actions = [p[s] for p in policies]
        if len(set(actions)) == 1:
            agree += 1
        else:
            disagree += 1
            print(
                f"    State {s}: "
                + ", ".join(f"{n}={a}" for n, a in zip(names, actions))
            )
    total = agree + disagree
    print(f"  Agreement: {agree}/{total} states " f"({100 * agree / total:.1f}%)")


# run part 2


def run_part2():
    env = GridWorld()
    timings = {}

    print("PART 2 - GRID WORLD MDP & REINFORCEMENT LEARNING")
    print(f"\n  Grid: {GRID_SIZE}x{GRID_SIZE}  |  Start: {START}  |  " f"Goal: {GOAL}")
    print(f"  Roadblocks: {ROADBLOCKS}")
    print(f"  γ={GAMMA}  α={ALPHA}  ε={EPSILON}")

    # 1a
    print("\n" + "─" * 60)
    print("  Task 1a - Value Iteration (deterministic transitions)")
    print("─" * 60)
    t0 = time.perf_counter()
    V_vi, pi_vi = value_iteration(env)
    timings["value_iteration"] = time.perf_counter() - t0
    print_value_function(V_vi, "Value Function (Value Iteration)")
    print_policy(pi_vi, "Optimal Policy (Value Iteration)")

    # 1b
    print("\n" + "─" * 60)
    print("  Task 1b - Policy Iteration (deterministic transitions)")
    print("─" * 60)
    t0 = time.perf_counter()
    V_pi, pi_pi = policy_iteration(env)
    timings["policy_iteration"] = time.perf_counter() - t0
    print_value_function(V_pi, "Value Function (Policy Iteration)")
    print_policy(pi_pi, "Optimal Policy (Policy Iteration)")

    # compare VI and PI
    print("\n  >> Comparing Value Iteration vs Policy Iteration:")
    compare_policies([pi_vi, pi_pi], ["VI", "PI"])

    # task 2
    print("\n" + "─" * 60)
    print("  Task 2 — Monte Carlo Control (stochastic, ε-greedy)")
    print("─" * 60)
    t0 = time.perf_counter()
    Q_mc, pi_mc, ep_mc = monte_carlo_control(env)
    timings["monte_carlo"] = time.perf_counter() - t0
    V_mc = q_to_v(Q_mc, env.states)
    print_value_function(V_mc, "Value Function (Monte Carlo)")
    print_policy(pi_mc, "Learned Policy (Monte Carlo)")

    # task 3
    print("\n" + "─" * 60)
    print("  Task 3 — Q-Learning (stochastic, ε-greedy)")
    print("─" * 60)
    t0 = time.perf_counter()
    Q_ql, pi_ql, ep_ql = q_learning(env)
    timings["q_learning"] = time.perf_counter() - t0
    V_ql = q_to_v(Q_ql, env.states)
    print_value_function(V_ql, "Value Function (Q-Learning)")
    print_policy(pi_ql, "Learned Policy (Q-Learning)")

    # comparison
    print("\n" + "─" * 60)
    print("  Overall Policy Comparison")
    print("─" * 60)
    compare_policies([pi_vi, pi_pi, pi_mc, pi_ql], ["VI", "PI", "MC", "QL"])

    print("\n  Runtime Summary")
    print("  " + "-" * 46)
    print(f"  Value Iteration:   {timings['value_iteration']:.4f} s")
    print(f"  Policy Iteration:  {timings['policy_iteration']:.4f} s")
    print(f"  Monte Carlo:       {timings['monte_carlo']:.4f} s")
    print(f"  Q-Learning:        {timings['q_learning']:.4f} s")

    return {
        "value_iteration": (V_vi, pi_vi),
        "policy_iteration": (V_pi, pi_pi),
        "monte_carlo": (Q_mc, pi_mc, ep_mc),
        "q_learning": (Q_ql, pi_ql, ep_ql),
    }


if __name__ == "__main__":
    run_part2()
