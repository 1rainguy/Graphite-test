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
try:
    from graphite.data.dataset_utils import load_default_dataset
except Exception:
    def load_default_dataset(*args, **kwargs):
        return None
try:
    from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
except Exception:
    def geom_edges(x):
        return x
    def euc_2d_edges(x):
        return x
    def man_2d_edges(x):
        return x
import numpy as np
import time
import asyncio
import random


class NearestNeighbourMultiSolver:
    def __init__(self):
        pass

    def solve(self, formatted_problem)->List[int]:
        def split_into_sublists(original_list, n_salesmen):
            n = len(original_list)
            sublist_size = n // n_salesmen  # Size of each sublist
            remainder = n % n_salesmen       # Remainder to distribute

            sublists = []
            start_index = 0

            for i in range(n_salesmen):
                # Add 1 to the size for the first 'remainder' sublists
                if i < remainder:
                    size = sublist_size + 1
                else:
                    size = sublist_size
                    
                sublists.append(original_list[start_index:start_index + size])
                start_index += size

            return sublists

        # naively apply greedy solution and compute total tour length
        m = formatted_problem.n_salesmen
        distance_matrix = formatted_problem.edges
        n = len(distance_matrix[0])
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
        # Naive split into m evenly sized sublists
        tours = split_into_sublists(route[1:], formatted_problem.n_salesmen)
        closed_tours = [[0] + tour + [0] for tour in tours]
        return closed_tours
    
    