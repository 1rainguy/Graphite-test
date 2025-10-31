# The MIT License (MIT)

from typing import List
import numpy as np
import time

class GreedyCycleSolver:
    """
    Greedy cycle heuristic: start with a small cycle and repeatedly insert the
    city whose insertion increases the tour length the least (greedy insertion).
    Distinct from cheapest insertion by seeding a small initial cycle.
    """

    def __init__(self, time_limit: int = 100):
        self.time_limit = time_limit

    def solve(self, formatted_problem) -> List[int]:
        dist = formatted_problem
        n = len(dist)
        start_time = time.time()
        if n <= 2:
            return list(range(n)) + [0]

        # Seed with a triangle: choose two farthest from 0
        i0 = 0
        j = int(np.argmax(dist[i0]))
        k = int(np.argmax(dist[j]))
        tour = [i0, j, k]
        remaining = set(range(n)) - set(tour)

        # Insert nodes greedily by minimal insertion cost
        while remaining and (time.time() - start_time) < self.time_limit:
            best_increase = float('inf')
            best_city = None
            best_pos = None
            for city in list(remaining)[:min(len(remaining), 500)]:
                # try all insertion arcs
                for pos in range(len(tour)):
                    a = tour[pos]
                    b = tour[(pos + 1) % len(tour)] if pos + 1 < len(tour) else tour[0]
                    increase = dist[a][city] + dist[city][b] - dist[a][b]
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        best_pos = (pos + 1) % len(tour)
            if best_city is None:
                break
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)

        tour.append(tour[0])
        return tour

    


