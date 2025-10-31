import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import SimulatedAnnealingSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob

PROBLEM_NUM = 10


def main():
    solver = SimulatedAnnealingSolver(time_limit=10)

    os.makedirs('results', exist_ok=True)
    out_path = 'results/simulated_annealing_results.csv'
    if os.path.exists(out_path):
        os.remove(out_path)

    for problem_id in range(PROBLEM_NUM):
        coords = read_prob(f'problems/{problem_id}.prob')
        dist = geom_edges(coords)
        route = solver.solve(dist)
        total_length = get_tour_length(route, dist)
        with open(out_path, 'a') as f:
            f.write(f"{problem_id},{total_length}\n")
        print(f'Problem {problem_id} route length: {total_length}')


if __name__ == '__main__':
    main()

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import SimulatedAnnealingSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob


def main():
    solver = SimulatedAnnealingSolver(time_limit=10)

    coords = read_prob(os.path.join(PROJECT_ROOT, 'problems', '3.prob'))
    dist = geom_edges(coords)
    route = solver.solve(dist)
    total_length = get_tour_length(route, dist)
    print('SimulatedAnnealingSolver route length:', total_length)


if __name__ == '__main__':
    main()


