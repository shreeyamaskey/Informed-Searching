import heapq

import numpy as np
from heapq import heappush, heappop
from animation import draw
import argparse

class Node():
    """
    cost_from_start - the cost of reaching this node from the starting node
    state - the state (row,col)
    parent - the parent node of this node, default as None
    """
    def __init__(self, state, cost_from_start, parent = None):
        self.state = state
        self.parent = parent
        self.cost_from_start = cost_from_start


class EightPuzzle():
    
    def __init__(self, start_state, goal_state, method, algorithm, array_index):
        self.start_state = start_state
        self.goal_state = goal_state
        self.visited = [] # state
        self.method = method
        self.algorithm = algorithm
        self.m, self.n = start_state.shape 
        self.array_index = array_index
        

    def goal_test(self, current_state):
        # your code goes here:
        return np.array_equal(current_state, self.goal_state)

    def get_cost(self, current_state, next_state):
        # your code goes here:
        return 1

    def get_successors(self, state):
        # your code goes here:
        successors = []

        # getting the position of where the empty spot is (0)
        positions = np.where(state == 0)
        # getting the specific row and column from the positions list to get the exact integers
        emptyrow, emptycol = positions[0][0], positions[1][0]

        def boundaries(coordinate, shift):
            rows, cols = state.shape
            if coordinate + shift not in range(rows):
                return True

        for moves in range(4):
            if moves % 2 == 0:
                shift = -1  # we go downwards or leftwards
            else:
                shift = 1  # we go upwards or rightwards

            if moves < 2:
                coordinate = emptyrow
            else:
                coordinate = emptycol

            if boundaries(coordinate, shift):
                continue

            temp = np.copy(state)
            if moves < 2:
                element = state[emptyrow + shift, emptycol]
                elementpos = np.where(state == element)
                temp[emptyrow, emptycol] = element
                temp[elementpos[0][0], elementpos[1][0]] = 0  # swapping the place of the empty spot and successor

            else:
                element = state[emptyrow, emptycol + shift]
                elementpos = np.where(state == element)
                temp[emptyrow, emptycol] = element
                temp[elementpos[0][0], elementpos[1][0]] = 0

            successors.append(temp)

        return successors

    # heuristics function
    def heuristics(self, state):
        # your code goes here:
        value = 0
        if self.method == 'Manhattan':
            #so we get the position of 8 blocks and their position in the goal state
            for number in range(1, 9):
                # getting the current position
                current_positions = np.where(state == number)
                # getting the specific row and column from the positions list to get the exact integers
                current_row, current_col = current_positions[0][0], current_positions[1][0]
                # getting the goal position
                goal_positions = np.where(self.goal_state == number)
                goal_row, goal_col = goal_positions[0][0], goal_positions[1][0]
                value += (abs(goal_row - current_row) + abs(goal_col - current_col))
        else:
            for number in range(1, 9):
                # getting the current position
                current_positions = np.where(state == number)
                # getting the specific row and column from the positions list to get the exact integers
                current_row, current_col = current_positions[0][0], current_positions[1][0]
                # getting the goal position
                goal_positions = np.where(self.goal_state == number)
                goal_row, goal_col = goal_positions[0][0], goal_positions[1][0]

                in_goal_state = (abs(goal_row - current_row) + abs(goal_col - current_col))
                if in_goal_state != 0:
                    value += 1

        return value

    # priority of node 
    def priority(self, node):
        # use if-else here to take care of different type of algorithms
        # your code goes here:
        if self.algorithm == 'Greedy':
            return self.heuristics(node.state)
        elif self.algorithm == 'AStar':
            return node.cost_from_start + self.heuristics(node.state)
    
    # draw 
    def draw(self, node):
        path=[]
        while node.parent:
            path.append(node.state)
            node = node.parent
        path.append(self.start_state)

        draw(path[::-1], self.array_index, self.algorithm, self.method)

    # solve it
    def solve(self):
        # use one framework to merge all five algorithms.
        # !!! In A* algorithm, you only need to return the first solution. 
        #     The first solution is in general possibly not the best solution, however, in this eight puzzle, 
        #     we can prove that the first solution is the best solution. 
        # your code goes here:    
        fringe = [] # node
        state = self.start_state.copy() # use copy() to copy value instead of reference
        node = Node(state, 0, None)
        self.visited.append(state)

        # the maximum depth cut off for dfs
        max_depth = 15
        depth = 0

        # in order to break the tie in priority queue
        tie_breaker = 0

        if self.goal_test(state):
            return state

        if self.algorithm == 'Depth-Limited-DFS':
            fringe.append(node)
            depth += 1

        elif self.algorithm == 'BFS':
            # push it into a queue
            fringe.insert(0, node)

        elif self.algorithm == 'Greedy':
            # push it into a queue
            item = (self.priority(node), tie_breaker, node)
            heapq.heappush(fringe, item)
        elif self.algorithm == 'AStar':
            # push it into a queue
            item = (self.priority(node), tie_breaker, node)
            heapq.heappush(fringe, item)

        while fringe:
            if self.algorithm == 'BFS':
                current = fringe.pop()
            elif self.algorithm == 'Depth-Limited-DFS':
                current = fringe.pop(0)
            elif self.algorithm == 'Greedy':
                current = heapq.heappop(fringe)
                current = current[len(current) - 1]
            elif self.algorithm == 'AStar':
                current = heapq.heappop(fringe)
                current = current[len(current) - 1]
            successors = self.get_successors(current.state)
            for next_state in successors:
                # to count that state is not in visited
                false_counter = 0
                for visited_state in self.visited:
                    if np.array_equal(visited_state, next_state) is False:
                        #run for everything in self.visited
                        false_counter += 1

                if false_counter == len(self.visited):
                    # create a node for next_state
                    next_cost = current.cost_from_start + self.get_cost(current.state, next_state)
                    next_node = Node(next_state, next_cost, current)
                    self.visited.append(next_state)

                    if self.goal_test(next_state) is True:
                        self.draw(next_node)
                        return next_state
                    if self.algorithm == 'BFS':
                        fringe.insert(0, next_node)
                    elif self.algorithm == 'Depth-Limited-DFS':
                        fringe.append(next_node)
                        depth += 1
                        if depth == max_depth:
                            continue
                    elif self.algorithm == 'Greedy':
                        tie_breaker += 1
                        item = (self.priority(next_node), tie_breaker, next_node)
                        heapq.heappush(fringe, item)
                    elif self.algorithm == 'AStar':
                        tie_breaker += 1
                        item = (self.priority(next_node), tie_breaker, next_node)
                        heapq.heappush(fringe, item)


if __name__ == "__main__":
    
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])
    start_arrays = [np.array([[1,2,0],[3,4,6],[7,5,8]]),
                    np.array([[8,1,3],[4,0,2],[7,6,5]])]
    methods = ["Hamming", "Manhattan"]
    algorithms = ['Depth-Limited-DFS', 'BFS', 'Greedy', 'AStar']
    
    parser = argparse.ArgumentParser(description='eight puzzle')

    parser.add_argument('-array', dest='array_index', required = True, type = int, help='index of array')
    parser.add_argument('-method', dest='method_index', required = True, type = int, help='index of method')
    parser.add_argument('-algorithm', dest='algorithm_index', required = True, type = int, help='index of algorithm')

    args = parser.parse_args()

    # Example:
    # Run this in the terminal using array 0, method Hamming, algorithm AStar:
    #     python eight_puzzle.py -array 0 -method 0 -algorithm 3
    game = EightPuzzle(start_arrays[args.array_index], goal, methods[args.method_index], algorithms[args.algorithm_index], args.array_index)
    game.solve()