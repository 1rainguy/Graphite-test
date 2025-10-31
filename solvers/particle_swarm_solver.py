# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

from typing import List
import numpy as np
import time
import random

class ParticleSwarmSolver:
    """
    Particle Swarm Optimization for TSP - simplified, synchronous version.
    """
    
    def __init__(self, time_limit: int = 100, n_particles: int = 50, n_iterations: int = 1000,
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        self.time_limit = time_limit
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def solve(self, formatted_problem) -> List[int]:
        dist = formatted_problem
        n = len(dist)
        start_time = time.time()
        if n <= 2:
            return list(range(n)) + [0]
        particles = self._initialize_particles(n, dist, start_time)
        if not particles:
            return list(range(n)) + [0]
        global_best_pos = min(particles, key=lambda p: p['best_cost'])['best_position'][:]
        global_best_cost = self._tour_cost(global_best_pos, dist)
        for iteration in range(self.n_iterations):
            if (time.time() - start_time) >= self.time_limit:
                break
            for particle in particles:
                if (time.time() - start_time) >= self.time_limit:
                    break
                self._update_particle(particle, global_best_pos)
                current_cost = self._tour_cost(particle['position'], dist)
                if current_cost < particle['best_cost']:
                    particle['best_position'] = particle['position'][:]
                    particle['best_cost'] = current_cost
                if current_cost < global_best_cost:
                    global_best_pos = particle['position'][:]
                    global_best_cost = current_cost
            if iteration % 50 == 0:
                self._adapt_parameters(iteration, self.n_iterations)
        return global_best_pos + [global_best_pos[0]]

    def _initialize_particles(self, n: int, dist: np.ndarray, start_time: float) -> List[dict]:
        particles = []
        for _ in range(min(5, self.n_particles // 10)):
            if (time.time() - start_time) >= self.time_limit:
                break
            tour = list(range(n))
            random.shuffle(tour)
            particles.append(self._create_particle(tour))
        while len(particles) < self.n_particles and (time.time() - start_time) < self.time_limit:
            tour = list(range(n))
            random.shuffle(tour)
            particles.append(self._create_particle(tour))
        return particles

    def _create_particle(self, position: List[int]) -> dict:
        n = len(position)
        velocity = np.random.uniform(-1, 1, n)
        return {
            'position': position[:],
            'velocity': velocity,
            'best_position': position[:],
            'best_cost': float('inf'),
        }

    def _update_particle(self, particle: dict, global_best: List[int]):
        n = len(particle['position'])
        r1, r2 = random.random(), random.random()
        cognitive = self.c1 * r1 * np.array(self._subtract_tours(particle['best_position'], particle['position']))
        social = self.c2 * r2 * np.array(self._subtract_tours(global_best, particle['position']))
        particle['velocity'] = self.w * np.array(particle['velocity']) + cognitive + social
        particle['position'] = self._add_velocity_to_tour(particle['position'], particle['velocity'])

    def _subtract_tours(self, tour1: List[int], tour2: List[int]) -> List[float]:
        n = len(tour1)
        diff = []
        for i in range(n):
            pos_in_tour2 = tour2.index(tour1[i])
            diff.append(pos_in_tour2 - i)
        return diff

    def _add_velocity_to_tour(self, tour: List[int], velocity: np.ndarray) -> List[int]:
        n = len(tour)
        rank = {city: idx for idx, city in enumerate(tour)}
        for city in rank:
            rank[city] = rank[city] + velocity[city]
        # Sort by updated rank
        return sorted(range(n), key=lambda city: rank[city])

    def _tour_cost(self, tour: List[int], dist: np.ndarray) -> float:
        if not dist or len(tour) < 2:
            return float('inf')
        total = 0.0
        for i in range(len(tour) - 1):
            total += dist[tour[i]][tour[i + 1]]
        total += dist[tour[-1]][tour[0]]
        return total

    def _adapt_parameters(self, iteration: int, max_iterations: int):
        progress = iteration / max_iterations
        self.w = 0.9 - 0.5 * progress
        self.c1 = 2.5 - 0.5 * progress
        self.c2 = 0.5 + 1.5 * progress
