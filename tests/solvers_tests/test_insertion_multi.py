import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import InsertionMultiSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob

PROBLEM_NUM = 10


class MultiDepotProblem:
    def __init__(self, edges, n_salesmen, depots):
        self.edges = edges
        self.n_salesmen = n_salesmen
        self.depots = depots


def tour_length_sum(tours, dist):
    total = 0.0
    for t in tours:
        total += get_tour_length(t, dist)
    return total


def main():
    solver = InsertionMultiSolver()

    os.makedirs('results', exist_ok=True)
    out_path = 'results/insertion_multi_results.csv'
    if os.path.exists(out_path):
        os.remove(out_path)

    for problem_id in range(PROBLEM_NUM):
        coords = read_prob(f'problems/{problem_id}.prob')
        dist = geom_edges(coords)
        n_salesmen = 2
        depots = [0] * n_salesmen
        prob = MultiDepotProblem(dist, n_salesmen=n_salesmen, depots=depots)
        tours = solver.solve(prob)
        total_length = tour_length_sum(tours, dist)
        with open(out_path, 'a') as f:
            f.write(f"{problem_id},{total_length}\n")
        print(f'Problem {problem_id} total multi-tour length: {total_length}')


if __name__ == '__main__':
    main()


