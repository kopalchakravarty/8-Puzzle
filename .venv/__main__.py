import heapq
import time
import sys
#from node import Node 

def is_solvable(initial_state_flat): # Check if the puzzle is solvable
    inversions = 0
    tiles = [tile for tile in initial_state_flat if tile != 0]
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions % 2 == 0

def get_user_input():

    print("Enter the initial state of the 8-puzzle on a 3 x 3 grid. Numbers 1-8 represent the puzzle state, use 0 for blank space.")
    print("Enter each row on a new line, with numbers separated by spaces.")

    initial_state = []
    while len(initial_state) < 3:
        row_num = len(initial_state) + 1
        try:
            row_input = input(f"Enter row {row_num}: ").strip().split()

            row = [int(x) for x in row_input]
            initial_state.append(row)

        except ValueError:
            print("Error: Please enter valid integers separated by spaces.")
            initial_state = [] # Reset if error

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            initial_state = [] # Reset if error

    # Check if solvable
    state = [tile for row in initial_state for tile in row]
    if not is_solvable(state):
        print("\nThe provided initial state is unsolvable. Abort")
        sys.exit(1)

    # Select heuristic if puzzle is valid
    print("\nSelect the search heuristic:")
    print("1. Uniform Cost Search")
    print("2. A* with Misplaced Tile Heuristic")
    print("3. A* with Manhattan Distance Heuristic")

    heuristic_map = {'1': 'uniform', '2': 'misplaced', '3': 'manhattan'}
    heuristic_choice_num = ''

    heuristic_choice_num = input("Enter your choice (1, 2, or 3): ").strip()
    return initial_state, heuristic_map[heuristic_choice_num]


def print_state(state):
    print("-" * 20)
    for row in state:
        print("|", end="")
        for tile in row:
            print(f" {tile if tile != 0 else ' '} ", end="|")
        print("\n" + "-" * 20)

# a-star function

if __name__ == "__main__":
    initial_state, heuristic = get_user_input()

    if initial_state and heuristic:
        print("\nInitial State:")
        print_state(initial_state)
# print execution path
