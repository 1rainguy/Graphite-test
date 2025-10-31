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
from .common_utils import nearest_neighbor, get_tour_length

class ThreeOptSolver:
    """
    3-opt TSP Solver implementation.
    
    This solver uses the 3-opt local search algorithm to improve TSP tours.
    It starts with a nearest neighbor solution and iteratively improves it
    by reconnecting three segments of the tour in different ways.
    """
    
    def __init__(self, time_limit: int = 100):
        """
        Initialize the 3-opt solver.
        
        Args:
            problem_types: List of problem types this solver can handle
            time_limit: Maximum time limit in seconds (default: 100)
        """
        self.time_limit = time_limit

    def solve(self, formatted_problem) -> List[int]:
        """
        Solve TSP using 3-opt algorithm.
        
        Args:
            formatted_problem: Distance matrix
            future_id: Future ID for tracking
            
        Returns:
            List of node indices representing the tour
        """
        distance_matrix = formatted_problem
        n = len(distance_matrix)

        current_tour = nearest_neighbor(dist=distance_matrix, start=0)
        
        start_time = time.time()
        improved = True
        
        while improved and (time.time() - start_time) < self.time_limit:
            improved = False
            best_distance = self._calculate_tour_distance(current_tour, distance_matrix)
            
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if (time.time() - start_time) >= self.time_limit:
                            break
                            
                        # Try all 3-opt reconnections
                        new_tours = self._get_three_opt_tours(current_tour, i, j, k)
                        
                        for new_tour in new_tours:
                            new_distance = self._calculate_tour_distance(new_tour, distance_matrix)
                            
                            if new_distance < best_distance:
                                current_tour = new_tour
                                best_distance = new_distance
                                improved = True
                                break
                                
                        if improved:
                            break
                            
                    if improved:
                        break
                        
        # Add the start node at the end to complete the cycle
        current_tour.append(current_tour[0])
        return current_tour

    def _get_three_opt_tours(self, tour, i, j, k):
        """Generate all possible 3-opt reconnections."""
        n = len(tour)
        tours = []
        
        # Original: A-B-C-D-E-F
        # After 3-opt cuts at i, j, k: A-B | C-D | E-F
        # We can reconnect in 7 different ways (excluding the original)
        
        # Tour 1: A-B-C-F-E-D
        tour1 = tour[:i] + tour[i:j] + tour[k:] + tour[j:k][::-1]
        tours.append(tour1)
        
        # Tour 2: A-B-E-D-C-F
        tour2 = tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:]
        tours.append(tour2)
        
        # Tour 3: A-B-E-F-C-D
        tour3 = tour[:i] + tour[j:k][::-1] + tour[k:] + tour[i:j]
        tours.append(tour3)
        
        # Tour 4: A-B-C-E-D-F
        tour4 = tour[:i] + tour[i:j] + tour[j:k][::-1] + tour[k:]
        tours.append(tour4)
        
        # Tour 5: A-B-D-C-E-F
        tour5 = tour[:i] + tour[i:j][::-1] + tour[j:k] + tour[k:]
        tours.append(tour5)
        
        # Tour 6: A-B-D-E-C-F
        tour6 = tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:]
        tours.append(tour6)
        
        # Tour 7: A-B-F-E-D-C
        tour7 = tour[:i] + tour[k:] + tour[j:k][::-1] + tour[i:j][::-1]
        tours.append(tour7)
        
        return tours

    def _calculate_tour_distance(self, tour, distance_matrix):
        """Calculate the total distance of a tour."""
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        return total_distance

    # Sync utility class; no extra metadata required
