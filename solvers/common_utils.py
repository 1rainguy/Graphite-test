from typing import List
import numpy as np
import time
import math

REF_EARTH_RADIUS = 6378.388

def nearest_neighbor(dist: np.ndarray, start: int = 0, start_time: float = None, hard_limit: float = None) -> List[int]:
    n = len(dist)
    if n == 0:
        return []
    tour = [start]
    visited = {start}
    current = start
    for _ in range(n - 1):
        if start_time is not None and hard_limit is not None:
            if (time.time() - start_time) >= hard_limit:
                break
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

def nearest_neighbor_subset(dist: np.ndarray, subset: List[int], start_time: float = None, hard_limit: float = None) -> List[int]:
    n = len(subset)
    if n <= 1:
        return subset
    start = subset[0]
    tour = [start]
    visited = {start}
    current = start
    for _ in range(n - 1):
        if start_time is not None and hard_limit is not None:
            if (time.time() - start_time) >= hard_limit:
                break
        nearest = None
        best_dist = float('inf')
        for city in subset:
            if city not in visited and dist[current][city] < best_dist:
                best_dist = dist[current][city]
                nearest = city
        if nearest is None:
            break
        tour.append(nearest)
        visited.add(nearest)
        current = nearest
    return tour

def two_opt_improve(solution: List[int], dist: np.ndarray, start_time: float = None, hard_limit: float = None, max_iterations: int = 20) -> List[int]:
    n = len(solution)
    improved_solution = solution[:]
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        if start_time is not None and hard_limit is not None:
            if (time.time() - start_time) >= hard_limit:
                break
        improved = False
        iterations += 1
        for i in range(1, n - 2):
            if start_time is not None and hard_limit is not None:
                if (time.time() - start_time) >= hard_limit:
                    break
            for j in range(i + 1, n):
                if start_time is not None and hard_limit is not None:
                    if (time.time() - start_time) >= hard_limit:
                        break
                a, b = improved_solution[i-1], improved_solution[i]
                c, d = improved_solution[j-1], improved_solution[j]
                if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-12:
                    improved_solution[i:j] = reversed(improved_solution[i:j])
                    improved = True
                    break
            if improved:
                break
    return improved_solution

def get_tour_length(tour: List[int], dist: np.ndarray) -> float:
    length = 0
    for i in range(len(tour) - 1):
        length += dist[tour[i]][tour[i + 1]]
    return length

def geom_edges(coords):
    '''
    Vectorized geom distance calculation using numpy.
    Requires lat_lon_array to be an Nx2 array where N is the number of nodes, 
    and the columns represent latitude and longitude respectively.
    '''
    n = len(coords)
    dist = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(i+1, n):
            lat2, lon2 = coords[j]
            # haversine formula:
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = (math.sin(dphi/2)**2 +
                 math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            d = REF_EARTH_RADIUS * c
            dist[i][j] = d
            dist[j][i] = d
    return dist

