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
import random
import time

class BeamSearchSolver:
    def __init__(self, time_limit: int = 10):
        self.time_limit = time_limit

    def solve(self, formatted_problem, beam_width:int=3)->List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        # Initialize the beam with the starting point (0) and a total distance of 0
        beam = [(0, [0], 0)]
        for _ in range(n - 1):
            if (time.time() - start_time) >= self.time_limit:
                break
            candidates = []

            # Expand each path in the beam
            for current_node, path, current_distance in beam:
                if (time.time() - start_time) >= self.time_limit:
                    break
                for next_node in range(n):
                    if next_node not in path:
                        new_path = path + [next_node]
                        new_distance = current_distance + distance_matrix[current_node][next_node]
                        candidates.append((next_node, new_path, new_distance))

            # Sort candidates by their current distance and select the top-k candidates (beam_width)
            candidates.sort(key=lambda x: x[2])
            beam = candidates[:min(beam_width, len(candidates))]

        # Complete or fallback if timed out
        final_candidates = []
        for current_node, path, current_distance in beam:
            if len(path) == n:
                final_distance = current_distance + distance_matrix[current_node][0]
                final_candidates.append((path + [0], final_distance))
            else:
                # Greedy-complete path quickly
                remaining = [i for i in range(n) if i not in path]
                curr = current_node
                total = current_distance
                comp_path = path[:]
                while remaining and (time.time() - start_time) < self.time_limit:
                    next_node = min(remaining, key=lambda j: distance_matrix[curr][j])
                    total += distance_matrix[curr][next_node]
                    comp_path.append(next_node)
                    curr = next_node
                    remaining.remove(next_node)
                # If timed out before visiting all, append the rest quickly
                if remaining:
                    for next_node in remaining:
                        total += distance_matrix[curr][next_node]
                        comp_path.append(next_node)
                        curr = next_node
                total += distance_matrix[curr][0]
                final_candidates.append((comp_path + [0], total))

        # Select best
        best_path, _ = min(final_candidates, key=lambda x: x[1])
        return best_path
    