# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
import time
from .common_utils import nearest_neighbor, two_opt_improve, get_tour_length

class TwoOptSolver():
    def __init__(self, time_limit: float =100):
        self.time_limit = time_limit
    
    def solve(self, formatted_problem)->List[int]:
        start_time = time.time()
        distance_matrix = formatted_problem
        n = len(distance_matrix)

        current = nearest_neighbor(dist=distance_matrix, start=0, start_time=start_time, hard_limit=self.time_limit)

        best = two_opt_improve(solution=current, dist=distance_matrix, start_time=start_time, hard_limit=self.time_limit, max_iterations=2000)
        total_length = get_tour_length(best, distance_matrix)
        print(f'Total length: {total_length}, time: {time.time() - start_time}')
        best.append(best[0])
        return best
