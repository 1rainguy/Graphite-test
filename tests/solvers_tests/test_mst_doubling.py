import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import MSTDoublingSolver
from solvers.common_utils import geom_edges, get_tour_length
from tests.utils import read_prob


def main():
    solver = MSTDoublingSolver(time_limit=5)

    coords = read_prob(os.path.join(PROJECT_ROOT, 'problems', '6.prob'))
    dist = geom_edges(coords)
    route = solver.solve(dist)
    total_length = get_tour_length(route, dist)
    print('MSTDoublingSolver route length:', total_length)


if __name__ == '__main__':
    main()


