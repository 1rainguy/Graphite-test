from typing import List
import numpy as np
import time
import random

class NearestNeighbourSolverVali:
    def __init__(self):
        pass

    def solve(self, formatted_problem) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        visited = [False] * n
        route = []

        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for node in range(n - 1):
            nearest_distance = np.inf
            nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])
            for j in range(n):
                if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            route.append(nearest_node)
            visited[nearest_node] = True
            current_node = nearest_node
        
        route.append(route[0])
        return route
