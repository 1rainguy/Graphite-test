# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List, Tuple
import numpy as np
import time
import random

class MemeticSolver:
    """
    Memetic Algorithm for TSP - quality-focused.
    
    Combines genetic algorithm with local search to find
    high-quality solutions within time limit.
    """
    
    def __init__(self, time_limit: int = 100, population_size: int = 80, 
                 generations: int = 1000, mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8, elite_size: int = 15,
                 local_search_rate: float = 0.4):
        self.time_limit = time_limit
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.local_search_rate = local_search_rate

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
                parent1 = self._tournament_selection(population, fitness_scores, tournament_size=3)
                parent2 = self._tournament_selection(population, fitness_scores, tournament_size=3)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._edge_recombination_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, distance_matrix)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, distance_matrix)
                
                if random.random() < self.local_search_rate:
                    child1 = self._local_search(child1, distance_matrix, start_time)
                if random.random() < self.local_search_rate:
                    child2 = self._local_search(child2, distance_matrix, start_time)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            fitness_scores = [self._calculate_fitness(individual, distance_matrix) for individual in population]
            
            if generation % 100 == 0:
                self._adapt_parameters(fitness_scores, generation)
        
        best_idx = np.argmin(fitness_scores)
        best_tour = population[best_idx]
        return best_tour + [best_tour[0]]

    def _initialize_population(self, n: int, dist: np.ndarray, start_time: float) -> List[List[int]]:
        population = []
        for _ in range(min(10, self.population_size // 5)):
            if (time.time() - start_time) >= self.time_limit:
                break
            nn_tour = self._simple_nn(dist)
            if nn_tour and len(nn_tour) == n:
                population.append(nn_tour)
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

    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float], tournament_size: int = 3) -> List[int]:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx][:]

    def _edge_recombination_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        n = len(parent1)
        edge_map1 = self._build_edge_map(parent1)
        edge_map2 = self._build_edge_map(parent2)
        child1 = self._build_child_from_edge_maps(edge_map1, edge_map2, n)
        child2 = self._build_child_from_edge_maps(edge_map2, edge_map1, n)
        return child1, child2

    def _build_edge_map(self, tour: List[int]) -> dict:
        n = len(tour)
        edge_map = {}
        for i in range(n):
            city = tour[i]
            prev_city = tour[i - 1]
            next_city = tour[(i + 1) % n]
            if city not in edge_map:
                edge_map[city] = set()
            edge_map[city].add(prev_city)
            edge_map[city].add(next_city)
        return edge_map

    def _build_child_from_edge_maps(self, edge_map1: dict, edge_map2: dict, n: int) -> List[int]:
        child = []
        remaining = set(range(n))
        current = random.randint(0, n - 1)
        child.append(current)
        remaining.remove(current)
        while remaining:
            connections = set()
            if current in edge_map1:
                connections.update(edge_map1[current])
            if current in edge_map2:
                connections.update(edge_map2[current])
            available = connections.intersection(remaining)
            if available:
                next_city = min(available, key=lambda city: len(remaining.intersection(
                    edge_map1.get(city, set()).union(edge_map2.get(city, set()))
                )))
            else:
                next_city = random.choice(list(remaining))
            child.append(next_city)
            remaining.remove(next_city)
            current = next_city
        return child

    def _mutate(self, individual: List[int], dist: np.ndarray) -> List[int]:
        mutated = individual[:] 
        mutation_type = random.choice(['swap', 'inversion', 'insertion', 'scramble'])
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
        elif mutation_type == 'scramble':
            i, j = sorted(random.sample(range(len(mutated)), 2))
            segment = mutated[i:j+1]
            random.shuffle(segment)
            mutated[i:j+1] = segment
        return mutated

    def _local_search(self, individual: List[int], dist: np.ndarray, start_time: float) -> List[int]:
        n = len(individual)
        improved = True
        iterations = 0
        max_iterations = 5
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

    def _adapt_parameters(self, fitness_scores: List[float], generation: int):
        diversity = np.std(fitness_scores)
        if diversity < 1000:
            self.mutation_rate = min(0.4, self.mutation_rate * 1.1)
            self.local_search_rate = min(0.8, self.local_search_rate * 1.05)
        else:
            self.mutation_rate = max(0.1, self.mutation_rate * 0.95)
            self.local_search_rate = max(0.2, self.local_search_rate * 0.98)
