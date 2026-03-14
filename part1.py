"""
SC3000/CZ3005 Lab Assignment 1 — Part 1: Graph Search Algorithms

Problem:
    Find optimal paths in a weighted directed graph from node "1" to node "50".

Data Files:
    G.json      : Graph adjacency list
    Coord.json  : Node coordinates
    Dist.json   : Edge distances
    Cost.json   : Edge energy costs

Tasks:
    1. Shortest Path - Dijkstra's Algorithm
       - Finds the path with  minimum total distance
       - Energy cost is calculated once the shortest path is found

    2. Uniform Cost Search with Energy Constraint
       - Finds the shortest path while ensuring total energy cost does not exceed the given energy budget

    3. A* Search with Energy Constraint
       - Uses a heuristic to guide the search toward the goal
       - Also enforces the energy budget constraint

Constraints:
    Start node: "1"
    Goal node: "50"
    Energy budget: 287932

Output:
    For each task, the program prints:
        - The path from start to goal
        - Total distance travelled
        - Total energy cost of the path
"""

import heapq
import math
import json
import time

# Data Loading

def load_json_file(filename):
    """
    Load and return data from a JSON file

    Args:
        filename (str): Path to the JSON file

    Returns:
        dict: Parsed JSON data as a Python dictionary
    """
    with open(filename, "r", encoding = "utf-8") as f:
        return json.load(f)
    
def load_data():
    """
    Load all required graph-related data files

    Returns:
        tuple:
            G (dict): Graph adjacency list
            Coord (dict): Node coordinates
            Dist (dict): Distance values for edges
            Cost (dict): Energy cost values for edges
    """
    G = load_json_file("G.json")
    Coord = load_json_file("Coord.json")
    Dist = load_json_file("Dist.json")
    Cost = load_json_file("Cost.json")
    return G, Coord, Dist, Cost

# Helper functions

def edge_key(u,v):
    """
    Generates the dictionary key used for edge-based lookups.

    Arguments:
        u (str): Source node
        v (str): Destination node

    Returns:
        str: Key representing the edge in the format 'u,v'
    """

    return f"{u},{v}"

def get_distance(Dist, u, v):
    """
    Retrieve the distance associated with an edge.

    Arguments:
        Dist (dict): Dictionary containing edge distances.
        u (str): Source node
        v (str): Destination node

    Returns:
        float: Distance value for edge (u, v)
    """
    return Dist[edge_key(u,v)]

def get_cost(Cost, u, v):
    """
    Retrieve the energy cost associated with an edge

    Arguments:
        Cost (dict): Dictionary containing energy costs
        u (str): Source node
        v (str): Destination node

    Returns:
        float: Energy cost for edge (u, v)
    """
    return Cost[edge_key(u,v)]

def heuristic(Coord, node, goal):
    """
    Compute the straight-line/Euclidean distance heuristic.

    Arguments:
        Coord (dict): Dictionary mapping nodes to (x, y) coordinates
        node (str): Current node
        goal (str): Goal node

    Returns:
        float: Estimated distance from the current node to the goal.
    """
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.hypot(x2-x1, y2-y1)

def reconstruct_path(parent, end_node):
    """
    Reconstruct a path from the start node to the given end node

    Arguments:
        parent (dict): Dictionary mapping nodes to their parent nodes
        end_node (str): The goal node

    Returns:
        list: Ordered list of nodes representing the path
    """
    path = []
    current = end_node

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

def compute_total_energy(path, Cost):
    """
    Compute the total energy cost for a given path

    Arguments:
        path (list): Sequence of nodes representing a path
        Cost (dict): Dictionary containing energy costs for edges

    Returns:
        float: Total energy cost of the path
    """

    total_energy = 0
    for i in range(len(path) - 1):
        total_energy += get_cost(Cost, path[i], path[i+1])

    return total_energy

def format_path(path):
    """
    Convert a list of nodes into a formatted string representation

    Arguments:
        path (list): Sequence of nodes

    Returns:
        str: Path formatted as 'node1->...->nodeN'
    """
    return "->".join(path)

#--------
# TASK 1
#--------

def dijkstra_shortest_path(G, Dist, Cost, start, goal):
    """
    Compute the shortest distance path using Dijkstra's algorithm.

    The algorithm expands nodes based on the smallest accumulated
    distance from the start node.
    """

    # min-heap: expanding the node with smallest distance so far
    pq = [(0, start)]

    # best_dist[node] = shortest distance found so far to this node
    best_dist = {start: 0}

    #parent[node] = previous node in the best path
    parent = {start: None}

    #visited = stores nodes whose shortest distance is finalised 
    visited = set()

    while pq:
        dist_so_far, u = heapq.heappop(pq)

        # if node already finalised, skip
        if u in visited:
            continue

        visited.add(u)

        #if we reached the goal, reconstruct and return the path
        if u == goal:
            path = reconstruct_path(parent, goal)
            total_energy = compute_total_energy(path, Cost)
            return path, dist_so_far, total_energy
        
        #try relaxing all outgoing edges from u
        for v in G[u]:
            if v in visited:
                continue

            new_dist = dist_so_far + get_distance(Dist, u, v)

            # if this path is better, then update
            if new_dist < best_dist.get(v, float("inf")):
                best_dist[v]= new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    return None, None, None

def print_result(task_name, path, total_distance, total_energy):
    print(f"\n{task_name}")

    if path is None:
        print("No path found.")
        return
    print(f"Shortest path: {format_path(path)} ")
    print(f"Shortest distance: {total_distance}")
    print(f"Total energy cost: {total_energy}")


#--------
# TASK 2
#--------

def ucs_with_energy(G, Dist, Cost, start, goal, budget):
    """
    Perform Uniform Cost Search (UCS) with an energy constraint.

    The algorithm searches for the minimum-distance path while ensuring
    the total energy cost does not exceed the specified budget. States
    that are dominated by better distance-energy combinations are pruned.
    """

    pq = [(0, start, 0)]  # (distance_so_far, node, energy_used)
    parent = {(start, 0): None}

    # For each node, keep a list of non-dominated states:
    # (energy_used, distance_so_far)
    best_states = {start: [(0, 0)]}

    while pq:
        dist_so_far, u, energy_used = heapq.heappop(pq)
        current_state = (u, energy_used)

        if u == goal:
            state_path = []
            s = current_state
            while s is not None:
                state_path.append(s)
                s = parent[s]
            state_path.reverse()

            node_path = [state[0] for state in state_path]
            return node_path, dist_so_far, energy_used

        for v in G[u]:
            edge_dist = get_distance(Dist, u, v)
            edge_cost = get_cost(Cost, u, v)

            new_dist = dist_so_far + edge_dist
            new_energy = energy_used + edge_cost

            if new_energy > budget:
                continue

            # Check whether this new state is dominated
            dominated = False
            if v in best_states:
                for e, d in best_states[v]:
                    if e <= new_energy and d <= new_dist:
                        dominated = True
                        break

            if dominated:
                continue

            # Remove old states that are dominated by this new one
            if v not in best_states:
                best_states[v] = []

            filtered = []
            for e, d in best_states[v]:
                if not (new_energy <= e and new_dist <= d):
                    filtered.append((e, d))
            filtered.append((new_energy, new_dist))
            best_states[v] = filtered

            next_state = (v, new_energy)
            parent[next_state] = current_state
            heapq.heappush(pq, (new_dist, v, new_energy))

    return None, None, None

# --------
# TASK 3
# --------

def astar_with_energy(G, Coord, Dist, Cost, start, goal, budget):
    """
    Perform A* search with an energy constraint.

    The algorithm uses a heuristic (Euclidean distance) to guide the search
    towards the goal while ensuring the energy cost does not exceed the
    specified budget. Dominated states are pruned to improve efficiency.
    
    """
    start_h = heuristic(Coord, start, goal)
    pq = [(start_h, 0, start, 0)]   # (f_score, g_score, node, energy_used)

    parent = {(start, 0): None}

    # best_states[node] = list of non-dominated (energy_used, distance_so_far) pairs
    best_states = {start: [(0, 0)]}

    while pq:
        f_score, dist_so_far, u, energy_used = heapq.heappop(pq)
        current_state = (u, energy_used)

        # Goal test
        if u == goal:
            state_path = []
            s = current_state
            while s is not None:
                state_path.append(s)
                s = parent[s]

            state_path.reverse()
            node_path = [state[0] for state in state_path]
            return node_path, dist_so_far, energy_used

        for v in G[u]:
            edge_dist = get_distance(Dist, u, v)
            edge_cost = get_cost(Cost, u, v)

            new_dist = dist_so_far + edge_dist
            new_energy = energy_used + edge_cost

            # Reject unfeasible states
            if new_energy > budget:
                continue

            # If we have already seen a better or equal state at v, skip this one
            dominated = False
            if v in best_states:
                for e, d in best_states[v]:
                    if e <= new_energy and d <= new_dist:
                        dominated = True
                        break

            if dominated:
                continue

            # Remove old states that are dominated by this new one
            if v not in best_states:
                best_states[v] = []

            new_list = []
            for e, d in best_states[v]:
                if not (new_energy <= e and new_dist <= d):
                    new_list.append((e, d))

            new_list.append((new_energy, new_dist))
            best_states[v] = new_list

            next_state = (v, new_energy)
            parent[next_state] = current_state

            new_f = new_dist + heuristic(Coord, v, goal)
            heapq.heappush(pq, (new_f, new_dist, v, new_energy))

    return None, None, None

def run_part1():
    """
    Execute all three tasks for the graph search problem

    Tasks performed:
        1. Dijkstra's algorithm for shortest distance path
        2. Uniform Cost Search with an energy constraint
        3. A* search with heuristic guidance and energy constraint

    The function loads graph data, runs each algorithm, and prints results
    """

    start = "1"
    goal = "50"
    energy_budget = 287932

    G, Coord, Dist, Cost = load_data()

    print("Graph nodes:", len(G))
    print("Coordinates:", len(Coord))
    print("Distances:", len(Dist))
    print("Energy costs:", len(Cost))

    # Task 1
    path1, dist1, energy1 = dijkstra_shortest_path(G, Dist, Cost, start, goal)
    print_result("Task 1 Result", path1, dist1, energy1)

    # Task 2
    path2, dist2, energy2 = ucs_with_energy(G, Dist, Cost, start, goal, energy_budget)
    print_result("Task 2 Result", path2, dist2, energy2)

    # Task 3
    path3, dist3, energy3 = astar_with_energy(G, Coord, Dist, Cost, start, goal, energy_budget)
    print_result("Task 3 Result", path3, dist3, energy3)
