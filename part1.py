import heapq
import math
import json

# Data Loading

def load_json_file(filename):
    #loading a dictionary from a JSON fil
    with open(filename, "r", encoding = "utf-8") as f:
        return json.load(f)
    
def load_data():
    #load the required dictionaries 
    G = load_json_file("G.json")
    Coord = load_json_file("Coord.json")
    Dist = load_json_file("Dist.json")
    Cost = load_json_file("Cost.json")
    return G, Coord, Dist, Cost

import heapq
import math

#helper functions

def edge_key(u,v):
    #dist and cost keys

    return f"{u},{v}"

def get_distance(Dist, u, v):
    # return distance of edge (u,v)
    return Dist[edge_key(u,v)]

def get_cost(Cost, u, v):
    #return energt cost of edge (u,v).
    return Cost[edge_key(u,v)]

def heuristic(Coord, node, goal):
    #straight-line distance from current node to goal node
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.hypot(x2-x1, y2-y1)

def reconstruct_path(parent, end_node):
    #reconstruct path from start to end using parent dictionary
    path = []
    current = end_node

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

def compute_total_energy(path, Cost):
    #given a full node path, sum the energy cost

    total_energy = 0
    for i in range(len(path) - 1):
        total_energy += get_cost(Cost, path[i], path[i+1])

    return total_energy

def format_path(path):
    return "->".join(path)

#--------
# TASK 1
#--------

def dijkstra_shortest_path(G, Dist, Cost, start, goal):

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
        
        #try relaxing all outgoing esges from u
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

            # Reject infeasible states
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
