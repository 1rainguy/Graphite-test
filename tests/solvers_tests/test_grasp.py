import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import GRASPSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob


def main():
    solver = GRASPSolver(time_limit=5, max_iterations=200)

    # Run over 10 problems and write results CSV
    results_path = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_path, exist_ok=True)
    out_csv = os.path.join(results_path, 'grasp_results.csv')
    with open(out_csv, 'w') as f:
        f.write('problem_id,total_length\n')

    for pid in range(10):
        coords = read_prob(os.path.join(PROJECT_ROOT, 'problems', f'{pid}.prob'))
        dist = geom_edges(coords)
        route = solver.solve(dist)
        total_length = get_tour_length(route, dist)
        with open(out_csv, 'a') as f:
            f.write(f"{pid},{total_length}\n")
        print(f'GRASP problem {pid} route length: {total_length}')


if __name__ == '__main__':
    main()


