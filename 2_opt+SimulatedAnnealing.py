import random
import math

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def tour_length(tour, points):
    return sum(euclidean(points[tour[i]], points[tour[i+1]]) for i in range(len(tour)-1))

def simulated_annealing(points, initial_tour=None, T0=1000, alpha=0.995, max_iter=10000):
    n = len(points)
    # Initial tour
    if initial_tour:
        current = initial_tour[:]
    else:
        current = list(range(n))
        random.shuffle(current)
        current.append(current[0])
    
    best = current[:]
    best_len = tour_length(best, points)
    T = T0
    
    for i in range(max_iter):
        # Generate neighbor: swap two cities
        a, b = random.sample(range(1, n), 2)
        neighbor = current[:]
        neighbor[a:b+1] = reversed(neighbor[a:b+1])
        
        delta = tour_length(neighbor, points) - tour_length(current, points)
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor[:]
            current_len = tour_length(current, points)
            if current_len < best_len:
                best = current[:]
                best_len = current_len
        
        # Cool down
        T *= alpha
        if T < 1e-8:
            break
    
    return best

points = [(0,0), (1,5), (5,2), (6,6), (8,3)]
best_tour = simulated_annealing(points)
print("Best tour:", best_tour)
