from deap import base, creator, tools, algorithms

import random
import math
from array import array

def tsp_genetic_algorithm(graph, population_size=100, generations=500):

    num_cities = len(graph)

    def evaluate(individual):

        return sum(graph[individual[i]][individual[i+1]] for i in range(num_cities - 1)) + graph[individual[-1]][individual[0]],

    # Guard against re-creating classes if this function is called multiple times
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("indices", random.sample, range(num_cities), num_cities)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)

    toolbox.register("mate", tools.cxOrdered)

    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=False)

    best_solution = tools.selBest(population, k=1)[0]

    return best_solution, evaluate(best_solution)[0]


def build_euclidean_graph(points):
    n = len(points)
    # Use compact C arrays to reduce memory for large n (e.g., 1000)
    graph = [array('d', [0.0] * n) for _ in range(n)]
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            d = math.hypot(x1 - x2, y1 - y2)
            graph[i][j] = d
            graph[j][i] = d
    return graph


def generate_random_points(n, seed=42, xrange=(0.0, 1000.0), yrange=(0.0, 1000.0)):
    rng = random.Random(seed)
    xmin, xmax = xrange
    ymin, ymax = yrange
    return [(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(n)]


if __name__ == "__main__":
    # Generate a 1000-node TSP instance
    num_nodes = 1000
    points = generate_random_points(num_nodes, seed=42)
    graph = build_euclidean_graph(points)
    # GA parameters may need tuning for your machine; these are reasonable defaults
    best_tour, best_cost = tsp_genetic_algorithm(graph, population_size=300, generations=400)
    # Print concise summary to avoid flooding the console
    preview = best_tour[:20]
    print("Nodes:", num_nodes)
    print("Tour length:", best_cost)
    print("Tour prefix (first 20 nodes):", preview, "...")