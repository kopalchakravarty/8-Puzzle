import heapq
import time
import sys
import math
from node import Node 

# Check if the puzzle is solvable. 
# An 8-puzzle problem is solvable as long as the number of inversions is even
# For an N-puzzle problem, this function can be modified with a modicum of effort to check for even/odd number of inversions
def is_solvable(initial_state): 
    inversions = 0
    tiles = [tile for tile in initial_state if tile != 0]
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] > tiles[j]:
                inversions += 1 # Increase the inversion count if the successor tile  < predecessor tile 
    return inversions % 2 == 0  # Return true if inversion count is even

# This function takes the user input
def get_user_input():

    print("Enter the initial state of the 8-puzzle on a 3 x 3 grid ")
    print("Numbers 1-8 represent the puzzle state, use 0 for blank space ")
    print("Numbers should be separated by spaces. Each row is a new line ")


    initial_state = []

    N = 8 # N is the number of tiles
    n = int(math.sqrt(N + 1)) # Calculate the dimension of the grid

    for i in range(n):
        row = i + 1 
        # Process the user input
        # Remove any erroneous blank spaces at the end using strip
        # Split the space separated user input into distinct list elements e.g  1 2 3 -> 1, 2, 3
        row_input = input(f"\nEnter row {row}: ").strip().split() 

        row = [int(x) for x in row_input] 
        initial_state.append(row) # Append the elements/row to the initial_state list

    # Check for solvable state. Abort execution if the state is not solvable
    state = [tile for row in initial_state for tile in row]
    if not is_solvable(state):
        print("\nThe provided puzzle is unsolvable. Aborting execution.")
        sys.exit(1)  # error
    
    # Input heuristic and map to the correct algorithm 
    print("\nSelect the search heuristic:")
    print("1. Uniform Cost Search")
    print("2. A* with Misplaced Tile Heuristic")
    print("3. A* with Manhattan Distance Heuristic")

    heuristic_map = {'1': 'uniform', '2': 'misplaced', '3': 'manhattan'}

    heuristic = input("Enter your choice (1, 2, or 3): ").strip()

    return initial_state, heuristic_map[heuristic]

# This function handles printing the "state" of the puzzle
def print_state(state):
    print("-" * 20)
    for row in state:
        print("| ", end="")
        for tile in row:
            print(f" {tile if tile != 0 else ' '} ", end="|")
        print("\n" + "-" * 20)

# A * search function
# Takes the initial state and the selected heuristic as input
def a_star_search(initial_state, heuristic):
    start_time = time.time() # Time when the search starts, used to calculate time taken by search

    root_node = Node(state=initial_state, heuristic=heuristic) # Initialize the root/first node by calling the Node class

    if root_node.state_tuple == Node.goal_state_tuple: # Check if the initial state is the goal state. The puzzle is already solved.
        print("Initial state is the goal state")
        return [], 0, 1, 0 # return Path, expanded_nodes, max_queue_size, solution_depth

    # This is our frontier. Use a heap to keep the node with the lowest f_cost at the front
    # We store tuples of (f_cost, node) so the heap sorts by f_cost (min heap). First node is with the lowest f_cost.
    nodes = [(root_node.f_cost, root_node)]

    # Turn the list into a heap
    heapq.heapify(nodes) 

    # Track the nodes we've visited and the min cost (g cost) to reach that node
    visited = {root_node.state_tuple: root_node.g_cost}

    # Initialize counters to track the metrics
    nodes_expanded = 0
    max_queue_size = 1

    # Loop while the frontier is not empty
    while nodes:
        max_queue_size = max(max_queue_size, len(nodes)) # Compare and update the max queue size
        f_cost, current_node = heapq.heappop(nodes)      # Pop the next best node to explore, lowest f_cost node

        # If the current state is the goal state, Stop the search. Return metrics and path
        if current_node.state_tuple == Node.goal_state_tuple:
            end_time = time.time() 
            path = current_node.get_path()
            depth = len(path)
            print(f"\n Goal reached in {end_time - start_time:.4f} seconds\n")
            return path, nodes_expanded, max_queue_size, depth, end_time - start_time
        
        # Else, continue to search by expanding the nodes/looking at its neighbours
        nodes_expanded += 1

        # Find all the possible next states/children from the current node. The function is defined in the Node class
        children = current_node.generate_children() 

        # Iterate over every child node
        for child in children:
            child_state_tuple = child.state_tuple
            child_g_cost = child.g_cost

            # If we have not visited the child state so far OR
            # The path cost (g) to get to the child node is cheaper via the current path, record the g_cost
            if child_state_tuple not in visited or child_g_cost < visited[child_state_tuple]:
                visited[child_state_tuple] = child_g_cost
                heapq.heappush(nodes, (child.f_cost, child)) # Add the child to the queue 
    
    # Loop ends when the frontier is empty.
    end_time = time.time()  # No solution found, stop the timer and end search
    print(f"No solution found after {end_time - start_time:.4f} sec. Search failed.")
    return None, nodes_expanded, max_queue_size, 0, end_time - start_time


# Driver code
if __name__ == "__main__":

    initial_state, heuristic = get_user_input() # Call the user input function


    if initial_state and heuristic:
        print("\n Initial State:")
        print_state(initial_state)

        print("\n Goal State:")
        print_state(Node.goal_state)

        print(f"\nBeginning search using {heuristic}: \n")
        print("*" * 20)

        # Begin the a-star seach 
        solution_path, nodes_expanded, max_queue_size, depth, time_taken = a_star_search(initial_state, heuristic)

        # Print the solution path and metrics if a solution exists and the path isn't empty
        if solution_path is not None:
            print("*" * 20)
            print(f" * Nodes Expanded: {nodes_expanded}")
            print(f" * Max Queue Size: {max_queue_size}")
            print(f" * Solution Depth: {depth}")
            print("-" * 20)
            
            print("\nSolution Path:\n")

            print_state(initial_state)
            for i, (action, state) in enumerate(solution_path):
                print(f"\nMove {i+1}: {action.upper()}\n")
                print_state(state)
                print("*" * 20)
            print("\n----- Goal Reached -----\n")
        else:
            print("\nNo solution could be found for the given initial state.")
            print("-" * 20)
            print(f" * Nodes Expanded: {nodes_expanded}")
            print(f" * Max Queue Size: {max_queue_size}")
            print("-" * 20)

