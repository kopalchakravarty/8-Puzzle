import math
import copy

class Node:
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]] 
    goal_positions = {}

    goal_state_tuple = tuple(map(tuple, goal_state))

    for row in range(3):
        for col in range(3):
            if goal_state[row][col] != 0:
                goal_positions[goal_state[row][col]] = (row, col)

    def __init__(self, state, parent=None, action=None, g_cost=0, heuristic='manhattan'):
        self.state = state

        self.state_tuple = tuple(map(tuple, self.state))
        self.parent = parent
        self.action = action
        self.g_cost = g_cost
        self.heuristic = heuristic
        self.h_cost = self.calculate_heuristic()
        self.f_cost = self.g_cost + self.h_cost

    def calculate_heuristic(self):
        if self.heuristic == 'uniform':
            return 0
        elif self.heuristic == 'misplaced':
            return self._misplaced_tiles()
        elif self.heuristic == 'manhattan':
            return self._manhattan_distance()

    def _misplaced_tiles(self):
        misplaced = 0
        for row in range(3):
            for col in range(3):
                current_tile = self.state[row][col]
                if current_tile != 0 and current_tile != Node.goal_state[row][col]:
                    misplaced += 1
        return misplaced

    def _manhattan_distance(self):
        total = 0
        for row in range(3):
            for col in range(3):
                current_tile = self.state[row][col]
                if current_tile != 0:
                    goal_row, goal_col = Node.goal_positions[current_tile]
                    total += abs(row - goal_row) + abs(col - goal_col)
        return total
    

    def find_blank(self):
        for r in range(3):
            for c in range(3):
                if self.state[r][c] == 0:
                    return r, c
        return None # Should not happen in a valid 8-puzzle

    def generate_children(self):
        children = []
        blank_row, blank_col = self.find_blank()

        if blank_row > 0: # Move UP
            new_state = copy.deepcopy(self.state)
            swap_row, swap_col = blank_row - 1, blank_col
            
            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp

            child_node = Node(state=new_state,
                              parent=self,
                              action='up', # Action taken was 'up'
                              g_cost=self.g_cost + 1, # Cost increases by 1 move
                              heuristic=self.heuristic)

            children.append(child_node) # Add to the list of children

        if blank_row < 2: # Move DOWN
            new_state = copy.deepcopy(self.state)
            swap_row, swap_col = blank_row + 1, blank_col

            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp
            
            child_node = Node(state=new_state,
                              parent=self,
                              action='down',
                              g_cost=self.g_cost + 1,
                              heuristic=self.heuristic)
            children.append(child_node)

        if blank_col > 0: # Move LEFT
            new_state = copy.deepcopy(self.state)
            swap_row, swap_col = blank_row, blank_col - 1

            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp

            child_node = Node(state=new_state,
                              parent=self,
                              action='left',
                              g_cost=self.g_cost + 1,
                              heuristic=self.heuristic)

            children.append(child_node)

        if blank_col < 2: # Move RIGHT
            new_state = copy.deepcopy(self.state)
            swap_row, swap_col = blank_row, blank_col + 1

            temp = new_state[blank_row][blank_col]
            new_state[blank_row][blank_col] = new_state[swap_row][swap_col]
            new_state[swap_row][swap_col] = temp
            
            child_node = Node(state=new_state,
                              parent=self,
                              action='right',
                              g_cost=self.g_cost + 1,
                              heuristic=self.heuristic)

            children.append(child_node)

        return children # Return valid children 

    def get_path(self):
        path = []
        current = self

        while current:
            path.insert(0, (current.action, current.state))
            current = current.parent
        return path[1:] # Remove the initial state's None action


    def __lt__(self, child):
        if self.f_cost != child.f_cost:
            return self.f_cost < child.f_cost
        return self.state_tuple < child.state_tuple

