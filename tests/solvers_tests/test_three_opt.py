import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import ThreeOptSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob


def main():
    solver = ThreeOptSolver(time_limit=10)

    coords = read_prob(os.path.join(PROJECT_ROOT, 'problems', '2.prob'))
    dist = geom_edges(coords)
    route = solver.solve(dist)
    total_length = get_tour_length(route, dist)
    print('ThreeOptSolver route length:', total_length)


if __name__ == '__main__':
    main()


