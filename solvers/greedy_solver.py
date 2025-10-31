# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
import numpy as np
import time
import random
import os

# bt is optional; not required for core logic

class NearestNeighbourSolver:
    def __init__(self):
        pass

    def solve(self, formatted_problem)->List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)

        visited = [False] * n
        route = []
        total_distance = 0

        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for node in range(n - 1):
            # Find the nearest unvisited neighbour
            nearest_distance = np.inf
            nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])# pre-set as random unvisited node
            for j in range(n):
                if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            # Move to the nearest unvisited node
            route.append(nearest_node)
            visited[nearest_node] = True
            total_distance += nearest_distance
            current_node = nearest_node
        
        # Return to the starting node
        total_distance += distance_matrix[current_node][route[0]]
        route.append(route[0])
        return route
        
