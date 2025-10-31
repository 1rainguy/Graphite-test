import random
import math

# Distance between points
def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# Total tour length
def tour_length(tour, points):
    return sum(euclidean(points[tour[i]], points[tour[i+1]]) for i in range(len(tour)-1))

# Generate initial population
def init_population(points, pop_size):
    n = len(points)
    population = []
    for _ in range(pop_size):
        tour = list(range(n))
        random.shuffle(tour)
        tour.append(tour[0])  # return to start
        population.append(tour)
    return population

# Order Crossover (OX)
def crossover(parent1, parent2):
    n = len(parent1) - 1
    a, b = sorted(random.sample(range(n), 2))
    child = [None]*n
    child[a:b] = parent1[a:b]
    ptr = 0
    for city in parent2:
        if city not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = city
    child.append(child[0])
    return child

# Mutation: swap two cities
def mutate(tour, rate=0.1):
    n = len(tour) - 1
    for i in range(1, n):
        if random.random() < rate:
            j = random.randint(1, n-1)
            tour[i], tour[j] = tour[j], tour[i]
    tour[-1] = tour[0]
    return tour

# Selection: tournament
def select(population, points, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda t: tour_length(t, points))
    return selected[0]

# Genetic Algorithm
def genetic_tsp(points, pop_size=50, generations=200, mutation_rate=0.1):
    population = init_population(points, pop_size)
    for g in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = select(population, points)
            p2 = select(population, points)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop
    # Return best tour
    best = min(population, key=lambda t: tour_length(t, points))
    return best


points = [(0,0), (1,5), (5,2), (6,6), (8,3)]
best_tour = genetic_tsp(points)
print("Best tour:", best_tour)
