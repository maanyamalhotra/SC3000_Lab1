"""
SC3000/CZ3005 Lab Assignment 1 — Main Runner

Runs Part 1 (Graph Search) and Part 2 (Grid World MDP & RL).
"""

from part1 import run_part1
from part2 import run_part2


def main():

    print("Running Part 1...")
    run_part1()

    print("\nRunning Part 2...")
    run_part2()

    print("Done. All algorithms executed successfully.")


if __name__ == "__main__":
    main()
