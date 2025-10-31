# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Tuple
import numpy as np
import time
import random

class AdvancedGeneticSolver:
    """
    Advanced Genetic Algorithm for TSP - quality-focused.
    """
    
    def __init__(self, time_limit: int = 100, population_size: int = 100, 
                 generations: int = 1000, mutation_rate: float = 0.3,
                 crossover_rate: float = 0.8, elite_size: int = 20):
        self.time_limit = time_limit
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

    def solve(self, formatted_problem) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        population = self._initialize_population(n, distance_matrix, start_time)
        if not population:
            return list(range(n)) + [0]

        fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
        
        for generation in range(self.generations):
            if (time.time() - start_time) >= self.time_limit:
                break
            new_population = []
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            while len(new_population) < self.population_size:
                if (time.time() - start_time) >= self.time_limit:
                    break
                parent1 = self._tournament_selection(population, fitness_scores, tournament_size=5)
                parent2 = self._tournament_selection(population, fitness_scores, tournament_size=5)
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, distance_matrix)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, distance_matrix)
                if random.random() < 0.3:
                    child1 = self._local_search(child1, distance_matrix, start_time)
                if random.random() < 0.3:
                    child2 = self._local_search(child2, distance_matrix, start_time)
                new_population.extend([child1, child2])
            population = new_population[:self.population_size]
            fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
            if generation % 50 == 0:
                self._adapt_parameters(fitness_scores)
        best_idx = np.argmin(fitness_scores)
        best_tour = population[best_idx]
        return best_tour + [best_tour[0]]

    def _initialize_population(self, n: int, dist: np.ndarray, start_time: float) -> List[List[int]]:
        population = []
        for _ in range(min(5, self.population_size // 10)):
            if (time.time() - start_time) >= self.time_limit:
                break
            try:
                nn_tour = self._simple_nn(dist)
                if nn_tour and len(nn_tour) > 1:
                    population.append(nn_tour)
            except:
                pass
        while len(population) < self.population_size and (time.time() - start_time) < self.time_limit:
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        return population

    def _simple_nn(self, dist: np.ndarray) -> List[int]:
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

    def _calculate_fitness(self, individual: List[int], distance_matrix: np.ndarray) -> float:
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i]][individual[i + 1]]
        return total_distance

    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float], tournament_size: int = 5) -> List[int]:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx][:]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        child1 = [-1] * n
        child1[start:end] = parent1[start:end]
        remaining = [x for x in parent2 if x not in child1[start:end]]
        idx = 0
        for i in range(n):
            if child1[i] == -1:
                child1[i] = remaining[idx]
                idx += 1
        child2 = [-1] * n
        child2[start:end] = parent2[start:end]
        remaining = [x for x in parent1 if x not in child2[start:end]]
        idx = 0
        for i in range(n):
            if child2[i] == -1:
                child2[i] = remaining[idx]
                idx += 1
        return child1, child2

    def _mutate(self, individual: List[int], dist: np.ndarray) -> List[int]:
        mutated = individual[:]
        mutation_type = random.choice(['swap', 'inversion', 'insertion'])
        if mutation_type == 'swap':
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif mutation_type == 'inversion':
            i, j = sorted(random.sample(range(len(mutated)), 2))
            mutated[i:j+1] = reversed(mutated[i:j+1])
        elif mutation_type == 'insertion':
            i, j = random.sample(range(len(mutated)), 2)
            if i < j:
                mutated.insert(j, mutated.pop(i))
            else:
                mutated.insert(i, mutated.pop(j))
        return mutated

    def _local_search(self, individual: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        n = len(individual)
        improved = True
        iterations = 0
        max_iterations = 10
        while improved and iterations < max_iterations and (time.time() - start_time) < self.time_limit:
            improved = False
            for i in range(1, n - 2):
                if (time.time() - start_time) >= self.time_limit:
                    break
                for j in range(i + 1, n):
                    if (time.time() - start_time) >= self.time_limit:
                        break
                    a, b = individual[i-1], individual[i]
                    c, d = individual[j-1], individual[j]
                    if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-12:
                        individual[i:j] = reversed(individual[i:j])
                        improved = True
                        break
                if improved:
                    break
            iterations += 1
        return individual

    def _adapt_parameters(self, fitness_scores: List[float]):
        diversity = np.std(fitness_scores)
        if diversity < 1000:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
            self.crossover_rate = max(0.5, self.crossover_rate * 0.9)
        else:
            self.mutation_rate = max(0.1, self.mutation_rate * 0.9)
            self.crossover_rate = min(0.9, self.crossover_rate * 1.1)
