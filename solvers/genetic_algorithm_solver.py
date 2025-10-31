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
from .common_utils import nearest_neighbor
import numpy as np
import time
import random

class GeneticAlgorithmSolver:
    """
    Genetic Algorithm TSP Solver implementation.
    
    This solver uses a genetic algorithm to find good TSP solutions.
    It maintains a population of tours and evolves them through
    selection, crossover, and mutation operations.
    """
    
    def __init__(self, time_limit: int = 100, population_size: int = 50, 
                 generations: int = 100, mutation_rate: float = 0.1,
                 elite_size: int = 10):
        """Initialize the genetic algorithm solver."""
        self.time_limit = time_limit
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def solve(self, formatted_problem) -> List[int]:
        """
        Solve TSP using genetic algorithm.
        
        Args:
            formatted_problem: Distance matrix
            future_id: Future ID for tracking
            
        Returns:
            List of node indices representing the tour
        """
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        
        # Initialize population
        population = self._initialize_population(n)
        
        # Evaluate initial population
        fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
        
        start_time = time.time()
        
        for generation in range(self.generations):
            if (time.time() - start_time) >= self.time_limit:
                break
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                child1, child2 = self._crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Replace population
            population = new_population[:self.population_size]
            
            # Evaluate new population
            fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
        
        # Return best individual
        best_idx = np.argmin(fitness_scores)
        best_tour = population[best_idx]
        
        # Add the start node at the end to complete the cycle
        best_tour.append(best_tour[0])
        return best_tour

    def _initialize_population(self, n):
        """Initialize population with random tours."""
        population = []
        
        # Add some nearest neighbor solutions (synchronously via utility)
        for _ in range(min(5, self.population_size // 4)):
            dummy_edges = np.random.rand(n, n)
            dummy_edges = (dummy_edges + dummy_edges.T) / 2  # Make symmetric
            np.fill_diagonal(dummy_edges, 0)
            nn_tour = nearest_neighbor(dummy_edges, start=0)
            if nn_tour and len(nn_tour) > 1:
                population.append(nn_tour)
        
        # Fill rest with random tours
        while len(population) < self.population_size:
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        
        return population

    def _calculate_fitness(self, individual, distance_matrix):
        """Calculate fitness (inverse of tour distance)."""
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i]][individual[i + 1]]
        return total_distance

    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx][:]

    def _crossover(self, parent1, parent2):
        """Perform order crossover (OX)."""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        # Create child1
        child1 = [-1] * n
        child1[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        remaining = [x for x in parent2 if x not in child1[start:end]]
        idx = 0
        for i in range(n):
            if child1[i] == -1:
                child1[i] = remaining[idx]
                idx += 1
        
        # Create child2
        child2 = [-1] * n
        child2[start:end] = parent2[start:end]
        
        # Fill remaining positions from parent1
        remaining = [x for x in parent1 if x not in child2[start:end]]
        idx = 0
        for i in range(n):
            if child2[i] == -1:
                child2[i] = remaining[idx]
                idx += 1
        
        return child1, child2

    def _mutate(self, individual):
        """Perform swap mutation."""
        mutated = individual[:]
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    # Sync utility class; no extra metadata required
