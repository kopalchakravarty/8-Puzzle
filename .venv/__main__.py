import heapq
import time
import sys
from node import Node 

def is_solvable(initial_state): # Check if the puzzle is solvable
    inversions = 0
    tiles = [tile for tile in initial_state if tile != 0]
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions % 2 == 0

def get_user_input():

    print("Enter the initial state of the 8-puzzle on a 3 x 3 grid ")
    print("Numbers 1-8 represent the puzzle state, use 0 for blank space ")
    print("Numbers should be separated by spaces. Each row is a new line ")


    initial_state = []

    for i in range(3):
        row = i + 1
        row_input = input(f"\nEnter row {row}: ").strip().split()

        row = [int(x) for x in row_input]
        initial_state.append(row)


    state = [tile for row in initial_state for tile in row]
    if not is_solvable(state):
        print("\nThe provided puzzle is unsolvable. Aborting execution.")
        sys.exit(1)  # error
    
    # Select heuristic
    print("\nSelect the search heuristic:")
    print("1. Uniform Cost Search")
    print("2. A* with Misplaced Tile Heuristic")
    print("3. A* with Manhattan Distance Heuristic")

    heuristic_map = {'1': 'uniform', '2': 'misplaced', '3': 'manhattan'}

    heuristic = input("Enter your choice (1, 2, or 3): ").strip()

    return initial_state, heuristic_map[heuristic]


def print_state(state):
    print("-" * 20)
    for row in state:
        print("|", end="")
        for tile in row:
            print(f" {tile if tile != 0 else ' '} ", end="|")
        print("\n" + "-" * 20)

def a_star_search(initial_state, heuristic):
    start_time = time.time()

    root_node = Node(state=initial_state, heuristic=heuristic)

    if root_node.state_tuple == Node.goal_state_tuple:
        print("Initial state is the goal state")
        return [], 0, 1, 0 # Path, expanded, max_queue, depth

    nodes = [(root_node.f_cost, root_node)]
    heapq.heapify(nodes)

    visited = {root_node.state_tuple: root_node.g_cost}

    nodes_expanded = 0
    max_queue_size = 1

    while nodes:
        max_queue_size = max(max_queue_size, len(nodes))
        f_cost, current_node = heapq.heappop(nodes)

        if current_node.state_tuple == Node.goal_state_tuple:
            end_time = time.time()
            path = current_node.get_path()
            depth = len(path)
            print(f"Goal reached in {end_time - start_time:.4f} seconds")
            return path, nodes_expanded, max_queue_size, depth

        nodes_expanded += 1

        children = current_node.generate_children()

        for child in children:
            child_state_tuple = child.state_tuple
            child_g_cost = child.g_cost

            # If the child state has not been visited OR
            # If the child state has been visited, but this new path (via current_node) is shorter (lower g_cost) than the previously found path
            if child_state_tuple not in visited or child_g_cost < visited[child_state_tuple]:
                visited[child_state_tuple] = child_g_cost
                heapq.heappush(nodes, (child.f_cost, child))

    end_time = time.time()
    print(f"No solution found after {end_time - start_time:.4f} sec. Search failed.")
    return None, nodes_expanded, max_queue_size, 0


if __name__ == "__main__":

    initial_state, heuristic = get_user_input()

    if initial_state and heuristic:
        print("\n Initial State:")
        print_state(initial_state)

        print("\n Goal State:")
        print_state(Node.goal_state)

        solution_path, num_expanded, max_q_len, sol_depth = a_star_search(initial_state, heuristic)

        print(f"\n Beginning search using {heuristic}: \n")
        print("*" * 20)

        if solution_path is not None:
            print("\n Solution Found: ")
            print("-" * 20)

            print(f" -> Nodes Expanded: {num_expanded}")
            print(f" -> Max Queue Size: {max_q_len}")
            print(f" -> Solution Depth: {sol_depth}")
            print("-" * 20)

            print("\nSolution Path:")
            print_state(initial_state)
            for i, (action, state) in enumerate(solution_path):
                print(f"Move {i+1}: {action.upper()}")
                print_state(state)
            print("----- Goal Reached -----")
        else:
            print("\nNo solution could be found for the given initial state.")
            print("-" * 20)
            print(f" -> Nodes Expanded: {num_expanded}")
            print(f" -> Max Queue Size: {max_q_len}")
            print("-" * 20)

