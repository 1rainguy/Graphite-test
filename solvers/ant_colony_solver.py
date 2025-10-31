from typing import List
from .common_utils import nearest_neighbor, get_tour_length
from .greedy_solver import NearestNeighbourSolver
import numpy as np
import time
import random

class AntColonySolver:
    """
    Ant Colony Optimization for TSP - quality-focused.
    
    Uses artificial ants to find high-quality solutions by following
    pheromone trails and heuristic information.
    """
    
    def __init__(self, time_limit: int = 100, n_ants: int = 50, n_iterations: int = 1000,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.5, q0: float = 0.9):
        self.time_limit = time_limit
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0

    def solve(self, formatted_problem) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        start_time = time.time()

        if n <= 2:
            return list(range(n)) + [0]

        tau = np.ones((n, n)) * 0.1
        best_tour = None
        best_cost = float('inf')
        
        for iteration in range(self.n_iterations):
            if (time.time() - start_time) >= self.time_limit:
                break
            
            tours = []
            for ant in range(self.n_ants):
                if (time.time() - start_time) >= self.time_limit:
                    break
                tour = self._construct_solution(distance_matrix, tau, start_time)
                if tour is not None:
                    tours.append(tour)
            
            for tour in tours:
                cost = self._tour_cost(tour, distance_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour[:]
            
            tau = self._update_pheromones(tau, tours, distance_matrix)
            
            if iteration % 50 == 0:
                self._adapt_parameters(iteration, self.n_iterations)
        
        if best_tour is None:
            nn_solver = NearestNeighbourSolver()
            fallback_tour = nn_solver.solve(formatted_problem)
            if fallback_tour and len(fallback_tour) > 1:
                if fallback_tour[-1] == fallback_tour[0]:
                    fallback_tour = fallback_tour[:-1]
                return fallback_tour + [fallback_tour[0]]
            return list(range(n)) + [0]
        
        return best_tour + [best_tour[0]]

    def _construct_solution(self, dist: np.ndarray, tau: np.ndarray, start_time: float) -> List[int]:
        n = len(dist)
        if n < 2:
            return None
            
        tour = []
        unvisited = set(range(n))
        current = random.randint(0, n - 1)
        tour.append(current)
        unvisited.remove(current)
        
        while unvisited and (time.time() - start_time) < self.time_limit:
            next_city = self._select_next_city(current, unvisited, dist, tau)
            if next_city is None:
                break
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        return tour if len(tour) == n else None

    def _select_next_city(self, current: int, unvisited: set, dist: np.ndarray, tau: np.ndarray) -> int:
        if not unvisited:
            return None
            
        unvisited_list = list(unvisited)
        
        probabilities = []
        for city in unvisited_list:
            if dist[current][city] == 0:
                continue
            pheromone = tau[current][city] ** self.alpha
            heuristic = (1.0 / dist[current][city]) ** self.beta
            probabilities.append(pheromone * heuristic)
        
        if not probabilities:
            return random.choice(unvisited_list)
        
        if random.random() < self.q0:
            max_idx = np.argmax(probabilities)
            return unvisited_list[max_idx]
        else:
            probabilities = np.array(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            return np.random.choice(unvisited_list, p=probabilities)

    def _update_pheromones(self, tau: np.ndarray, tours: List[List[int]], dist: np.ndarray) -> np.ndarray:
        n = len(tau)
        
        tau *= (1 - self.rho)
        
        for tour in tours:
            if len(tour) < 2:
                continue
            tour_cost = self._tour_cost(tour, dist)
            if tour_cost > 0:
                delta_tau = 1.0 / tour_cost
                for i in range(len(tour) - 1):
                    tau[tour[i]][tour[i + 1]] += delta_tau
                tau[tour[-1]][tour[0]] += delta_tau
        
        tau = np.clip(tau, 0.1, 10.0)
        
        return tau

    def _tour_cost(self, tour: List[int], dist: np.ndarray) -> float:
        if len(tour) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += dist[tour[i]][tour[i + 1]]
        return cost

    def _adapt_parameters(self, iteration: int, max_iterations: int):
        progress = iteration / max_iterations
        
        self.q0 = 0.5 + 0.4 * progress
        
        self.rho = 0.3 + 0.4 * progress
