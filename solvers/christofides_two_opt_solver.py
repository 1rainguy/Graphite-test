# The MIT License (MIT)

from typing import List
from .christofides_solver import ChristofidesSolver
from .common_utils import two_opt_improve
import time

class ChristofidesTwoOptSolver:
    """
    Wrapper: run Christofides to build a tour, then apply 2-opt within time limit.
    """

    def __init__(self, time_limit: int = 100):
        self.time_limit = time_limit

    def solve(self, formatted_problem) -> List[int]:
        start_time = time.time()
        dist = formatted_problem
        n = len(dist)
        if n <= 2:
            return list(range(n)) + [0]

        base = ChristofidesSolver(time_limit=max(1, int(self.time_limit * 0.7)))
        tour = base.solve(formatted_problem)
        if tour and len(tour) > 1 and tour[-1] == tour[0]:
            tour = tour[:-1]

        tour = two_opt_improve(solution=tour, dist=dist, start_time=start_time, hard_limit=self.time_limit, max_iterations=20)
        tour.append(tour[0])
        return tour

    


