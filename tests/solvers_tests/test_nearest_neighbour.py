import os
import sys
import importlib
import asyncio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from tests.utils import ensure_dummy_solvers_package, read_prob, build_euclidean_dist, TestGraphV2Problem, run_solver


def main():
    project_root = PROJECT_ROOT
    ensure_dummy_solvers_package(project_root)

    coords = read_prob(os.path.join(project_root, 'problems', '0.prob'), limit=50)
    dist = build_euclidean_dist(coords)
    problem = TestGraphV2Problem(dist)

    mod = importlib.import_module('solvers.greedy_solver')
    SolverCls = getattr(mod, 'NearestNeighbourSolver')
    route = asyncio.run(run_solver(SolverCls, problem, timeout_seconds=5))
    print('NearestNeighbourSolver route length:', len(route) if route else 0)


if __name__ == '__main__':
    main()


