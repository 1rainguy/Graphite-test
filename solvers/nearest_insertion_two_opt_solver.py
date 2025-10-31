# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List
import numpy as np
import time
from .common_utils import two_opt_improve

class NearestInsertionTwoOptSolver:
    """
    Nearest Insertion + 2-opt for TSP.
    
    Nearest Insertion is similar to Cheapest Insertion but picks the unvisited node
    closest to any visited node. Then improves with 2-opt.
    """
    
    def __init__(self, time_limit: int = 100):
        self.time_limit = time_limit

    def solve(self, formatted_problem) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()
        
        effective_time_limit = self.time_limit * 0.95

        if n <= 2:
            route = list(range(n)) + [0]
            return route

        if n > 1000:
            tour = self._fallback_nn(distance_matrix)
        else:
            insertion_time_limit = effective_time_limit * 0.6
            tour = self._nearest_insertion(distance_matrix, start_time, insertion_time_limit)
            if tour is None or len(tour) < n:
                tour = self._fallback_nn(distance_matrix)

        if (time.time() - start_time) >= effective_time_limit:
            return tour + [tour[0]]

        if n <= 500:
            tour = self._two_opt_improve(tour, distance_matrix, start_time, effective_time_limit)
        return tour + [tour[0]]
    
    def _fallback_nn(self, dist):
        n = len(dist)
        tour = [0]
        visited = {0}
        current = 0
        for _ in range(n - 1):
            nearest = None
            best_dist = float('inf')
            for j in range(n):
                if j not in visited and dist[current][j] < best_dist:
                    best_dist = dist[current][j]
                    nearest = j
            if nearest is None:
                break
            tour.append(nearest)
            visited.add(nearest)
            current = nearest
        return tour

    def _nearest_insertion(self, dist: np.ndarray, start_time: float, time_limit: float) -> List[int]:
        n = len(dist)
        best_dist = float('inf')
        i0, j0 = 0, 1
        search_limit = min(n, 50)
        for i in range(search_limit):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(i + 1, min(i + 20, n)):
                if dist[i][j] < best_dist:
                    best_dist = dist[i][j]
                    i0, j0 = i, j
        
        tour = [i0, j0]
        remaining = set(range(n)) - set(tour)
        check_counter = 0

        while remaining:
            check_counter += 1
            if check_counter % 10 == 0 and (time.time() - start_time) >= time_limit:
                break
                
            best_dist_to_tour = float('inf')
            best_node = None
            remaining_list = list(remaining)[:min(50, len(remaining))]
            for node in remaining_list:
                if check_counter % 5 == 0 and (time.time() - start_time) >= time_limit:
                    break
                min_dist_to_node = min([dist[node][t] for t in tour])
                if min_dist_to_node < best_dist_to_tour:
                    best_dist_to_tour = min_dist_to_node
                    best_node = node

            if best_node is None:
                if remaining:
                    tour.extend(list(remaining))
                    break
                break

            best_pos = 0
            best_cost = float('inf')
            for k in range(len(tour)):
                a, b = tour[k], tour[(k + 1) % len(tour)]
                cost = dist[a][best_node] + dist[best_node][b] - dist[a][b]
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (k + 1) % len(tour)
            
            tour.insert(best_pos, best_node)
            remaining.remove(best_node)
        
        return tour

    def _two_opt_improve(self, tour: List[int], dist: np.ndarray, start_time: float, time_limit: float) -> List[int]:
        return two_opt_improve(solution=tour, dist=dist, start_time=start_time, hard_limit=time_limit, max_iterations=5)

