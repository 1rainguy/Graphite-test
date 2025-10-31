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
import itertools
import time

class DPSolver:
    def __init__(self):
        pass

    def solve(self, formatted_problem)->List[int]:
        distance_matrix = np.array(formatted_problem, dtype=float)
        if not self.is_solvable(distance_matrix):
            return []
        n = len(distance_matrix)
        if n <= 1:
            return [0] if n == 1 else []
        # Held-Karp DP starting from 0
        dp = {(1 << i, i): (distance_matrix[0][i], [0, i]) for i in range(1, n)}
        for r in range(2, n):
            for subset in itertools.combinations(range(1, n), r):
                bits = 0
                for b in subset:
                    bits |= 1 << b
                for j in subset:
                    prev_bits = bits & ~(1 << j)
                    best_cost = float('inf')
                    best_path = []
                    for k in subset:
                        if k == j:
                            continue
                        state = (prev_bits, k)
                        if state in dp:
                            cost, path = dp[state]
                            c = cost + distance_matrix[k][j]
                            if c < best_cost:
                                best_cost = c
                                best_path = path + [j]
                    dp[(bits, j)] = (best_cost, best_path)
        bits = (1 << n) - 2
        best_cost = float('inf')
        best_path = []
        for j in range(1, n):
            state = (bits, j)
            if state in dp:
                cost, path = dp[state]
                c = cost + distance_matrix[j][0]
                if c < best_cost:
                    best_cost = c
                    best_path = path + [0]
        return best_path
 
    def is_solvable(self, distance_matrix):
        # checks if any row or any col has only inf values
        distance_arr = np.array(distance_matrix).astype(np.float32)
        np.fill_diagonal(distance_arr, np.inf)
        rows_with_only_inf = np.all(np.isinf(distance_arr) | np.isnan(distance_arr), axis=1)
        has_row_with_only_inf = np.any(rows_with_only_inf)

        cols_with_only_inf = np.all(np.isinf(distance_arr) | np.isnan(distance_arr), axis=0)
        has_col_with_only_inf = np.any(cols_with_only_inf)
        return not has_col_with_only_inf and not has_row_with_only_inf
    
