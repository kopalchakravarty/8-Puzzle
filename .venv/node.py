import math
import copy

# This class represents a single state/arragnement in the puzzle search
# Each node knows its state, path, and its cost

N = 8 # Number of tiles
n = int(math.sqrt(N + 1)) # Grid size. Depends on the value of N

class Node:
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]] # Default goal state
    goal_positions = {}

    # Make the goal state into a tuple of tuples. 
    # Tuples are immutable. Python needs to make things 'hashable' before we can use them for comparision
    goal_state_tuple = tuple(map(tuple, goal_state))

    # Figure out the goal position for each tile, store in a dictionary
    for row in range(n):
        for col in range(n):
            if goal_state[row][col] != 0:
                goal_positions[goal_state[row][col]] = (row, col)

    # Initialize/Create a new Node object
    def __init__(self, state, parent=None, action=None, g_cost=0, heuristic='manhattan'):
        self.state = state

        self.state_tuple = tuple(map(tuple, self.state)) # Make the current state into a tuple of tuples. 
        self.parent = parent                             # The node that we came from to get to this state
        self.action = action                             # The action that got us to this state
        self.g_cost = g_cost                             # Path cost to get to this state
        self.heuristic = heuristic                       # Heuristic to use as selected by the user
        self.h_cost = self.calculate_heuristic()         # Calculate the h_cost 
        self.f_cost = self.g_cost + self.h_cost          # Calculate f_cost 


    # Function to calculate the h_cost
    def calculate_heuristic(self):
        if self.heuristic == 'uniform': # Uniform cost search is like A* with h=0
            return 0
        elif self.heuristic == 'misplaced': # Call the function
            return self._misplaced_tiles()
        elif self.heuristic == 'manhattan':
            return self._manhattan_distance() # Call the function

    # Calculate h_cost for misplaced tile heuristic
    def _misplaced_tiles(self):
        misplaced = 0
        # Iterate through every spot in the grid
        for row in range(n):
            for col in range(n):
                current_tile = self.state[row][col]
                # If the current tile is not the blank AND not in the right place, increment count by 1
                if current_tile != 0 and current_tile != Node.goal_state[row][col]:
                    misplaced += 1
        return misplaced # Return the total count of misplaced tiles for the current state

    def _manhattan_distance(self):
        total = 0
        # Iterate through every spot in the grid
        for row in range(n):
            for col in range(n):
                current_tile = self.state[row][col]
                # If the current tile is not the blank, calculate the manhattan distance from the goal state
                if current_tile != 0:
                    goal_row, goal_col = Node.goal_positions[current_tile] # Get the actual/expected coordinates of the current tile
                    total += abs(row - goal_row) + abs(col - goal_col)     # Calulate the distance betwwen the present and expected coordinates
        return total # Repeat for all tiles in the grid and return the total manhattan distance for the current state
    
    # Find the coordinates of the blank tile
    def find_blank(self):
        for r in range(n):
            for c in range(n):
                if self.state[r][c] == 0:
                    return r, c # Return its row and column
        return None 
    
    # Generate all possible next states/children nodes from the current state
    def generate_children(self):
        children = []
        blank_row, blank_col = self.find_blank() # Find where the blank tile is

        if blank_row > 0: # Move UP
            new_state = copy.deepcopy(self.state) # Make a copy of the current state and modigy it to get the new state
            swap_row, swap_col = blank_row - 1, blank_col # Swap with the tile directly above the blank space
            
            # Swap tiles
            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp

            # Create a new Node for this new state 
            node = Node(state=new_state,
                              parent=self,
                              action='up', # Action taken was 'up'
                              g_cost=self.g_cost + 1, # Cost increases by 1 move
                              heuristic=self.heuristic)

            children.append(node) # Add to the list of children

        if blank_row < 2: # Move DOWN
            new_state = copy.deepcopy(self.state) # Make a copy of the current state and modigy it to get the new state
            swap_row, swap_col = blank_row + 1, blank_col # Swap with the tile directly below the blank space

            # Swap tiles
            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp
            
            # Create a new Node for this new state 
            node = Node(state=new_state,
                              parent=self,
                              action='down',
                              g_cost=self.g_cost + 1,
                              heuristic=self.heuristic)
            
            children.append(node) # Add to the list of children

        if blank_col > 0: # Move LEFT
            new_state = copy.deepcopy(self.state)  # Make a copy of the current state and modigy it to get the new state
            swap_row, swap_col = blank_row, blank_col - 1 # Swap with the tile directly left of the blank space

            # Swap tiles
            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp

            # Create a new Node for this new state 
            node = Node(state=new_state,
                              parent=self,
                              action='left',
                              g_cost=self.g_cost + 1,
                              heuristic=self.heuristic)

            children.append(node) # Add to the list of children

        if blank_col < 2: # Move RIGHT
            new_state = copy.deepcopy(self.state) # Make a copy of the current state and modigy it to get the new state
            swap_row, swap_col = blank_row, blank_col + 1 # Swap with the tile directly right of the blank space

            # Swap tiles
            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp
            
            # Create a new Node for this new state 
            node = Node(state=new_state,
                              parent=self,
                              action='right',
                              g_cost=self.g_cost + 1,
                              heuristic=self.heuristic)

            children.append(node) # Add to the list of children

        return children # Return valid children 

    # Reconstruct the path from the initial state to the current node

    def get_path(self):
        path = []
        current = self

        # Keep looping until we reach the root node (with no parents)
        while current:
            path.insert(0, (current.action, current.state)) # Add the action that led to this node and the current state to the list
            current = current.parent 
        return path[1:] # Remove the (None, initial_state) element for the intital state node

    # Used by the Python heapq priority queue to sort the nodes
    # Nodes with lower f_cost are considered more important that nodes with higher f_cost
    def __lt__(self, child):
        if self.f_cost != child.f_cost:
            return self.f_cost < child.f_cost
        return self.state_tuple < child.state_tuple

