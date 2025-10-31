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
import random
import math
from .common_utils import nearest_neighbor, get_tour_length

class SimulatedAnnealingSolver:
    """
    Simulated Annealing TSP Solver implementation.
    
    This solver uses simulated annealing to find good TSP solutions.
    It starts with a random or nearest neighbor solution and uses
    temperature-based acceptance of worse solutions to escape local optima.
    """
    
    def __init__(self, time_limit: int = 100, initial_temp: float = 1000.0, 
                 cooling_rate: float = 0.95, min_temp: float = 0.1):
        """
        Initialize the simulated annealing solver.
        
        Args:
            problem_types: List of problem types this solver can handle
            time_limit: Maximum time limit in seconds (default: 100)
            initial_temp: Initial temperature for annealing (default: 1000.0)
            cooling_rate: Rate at which temperature decreases (default: 0.95)
            min_temp: Minimum temperature to stop annealing (default: 0.1)
        """
        self.time_limit = time_limit
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def solve(self, formatted_problem) -> List[int]:
        """
        Solve TSP using simulated annealing.
        
        Args:
            formatted_problem: Distance matrix
            future_id: Future ID for tracking
            
        Returns:
            List of node indices representing the tour
        """
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        current_tour = nearest_neighbor(distance_matrix, start=0)
        
        # Initialize best solution
        best_tour = current_tour[:]
        best_distance = self._calculate_tour_distance(current_tour, distance_matrix)
        current_distance = best_distance
        
        # Simulated annealing parameters
        temperature = self.initial_temp
        start_time = time.time()
        
        while temperature > self.min_temp and (time.time() - start_time) < self.time_limit:
            # Generate a neighbor solution using 2-opt
            neighbor_tour = self._generate_neighbor(current_tour, n)
            neighbor_distance = self._calculate_tour_distance(neighbor_tour, distance_matrix)
            
            # Calculate acceptance probability
            delta = neighbor_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_tour = neighbor_tour
                current_distance = neighbor_distance
                
                # Update best solution if improved
                if current_distance < best_distance:
                    best_tour = current_tour[:]
                    best_distance = current_distance
            
            # Cool down
            temperature *= self.cooling_rate
            
        # Add the start node at the end to complete the cycle
        best_tour.append(best_tour[0])
        return best_tour

    def _generate_neighbor(self, tour, n):
        """Generate a neighbor solution using 2-opt swap."""
        # Create a copy of the tour
        neighbor = tour[:]
        
        # Choose two random positions for 2-opt swap
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Perform 2-opt swap
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        
        return neighbor

    def _calculate_tour_distance(self, tour, distance_matrix):
        """Calculate the total distance of a tour."""
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        return total_distance

    # Sync utility class; no extra metadata required
