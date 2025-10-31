import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import TabuSearchSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob


def main():
    solver = TabuSearchSolver(time_limit=5, max_iterations=500)

    coords = read_prob(os.path.join(PROJECT_ROOT, 'problems', '7.prob'))
    dist = geom_edges(coords)
    route = solver.solve(dist)
    total_length = get_tour_length(route, dist)
    print('TabuSearchSolver route length:', total_length)


if __name__ == '__main__':
    main()


