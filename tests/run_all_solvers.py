import os
import time
import math
import asyncio
from typing import List, Tuple

import numpy as np
import sys

# Ensure project root (parent of tests/) is on sys.path for local imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Local imports
import importlib
from typing import Optional


SKIP_SOLVERS = {
    # External tool dependencies or very heavy
    'LKHSolver',
    'ConcordeSolver',
    'ConcordeHybridSolver',
    'HPNSolver',
}


def read_prob(path: str, limit: int = 150) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    # First line is a count, following lines are "lat lon"
    coords: List[Tuple[float, float]] = []
    for line in lines[1:]:
        try:
            a, b = line.split()
            coords.append((float(a), float(b)))
        except Exception:
            continue
        if len(coords) >= limit:
            break
    if len(coords) < 2:
        raise ValueError(f"Not enough coordinates in {path}")
    return np.array(coords, dtype=float)


def build_euclidean_dist(coords: np.ndarray) -> np.ndarray:
    # Great-circle not necessary; use Euclidean for test consistency
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist


class TestGraphV2Problem:
    def __init__(self, edges: np.ndarray):
        self.problem_type = "Metric TSP"
        self.n_nodes = edges.shape[0]
        self.edges = edges


def make_problem(dist: np.ndarray) -> TestGraphV2Problem:
    return TestGraphV2Problem(dist)


SOLVER_SPECS: List[Tuple[str, str]] = [
    ("solvers.greedy_solver", "NearestNeighbourSolver"),
    ("solvers.tsp_solver", "TSPSOLVER"),
    ("solvers.two_opt_solver", "TwoOptSolver"),
    ("solvers.three_opt_solver", "ThreeOptSolver"),
    ("solvers.simulated_annealing_solver", "SimulatedAnnealingSolver"),
    ("solvers.mst_doubling_solver", "MSTDoublingSolver"),
    ("solvers.lin_kernighan_solver", "LinKernighanSolver"),
    ("solvers.genetic_algorithm_solver", "GeneticAlgorithmSolver"),
    ("solvers.advanced_genetic_solver", "AdvancedGeneticSolver"),
    ("solvers.memetic_solver", "MemeticSolver"),
    ("solvers.grasp_solver", "GRASPSolver"),
    ("solvers.variable_neighborhood_solver", "VariableNeighborhoodSolver"),
    ("solvers.tabu_search_solver", "TabuSearchSolver"),
    ("solvers.nearest_insertion_two_opt_solver", "NearestInsertionTwoOptSolver"),
    ("solvers.evolution_strategies_solver", "EvolutionStrategiesSolver"),
    ("solvers.iterated_local_search_solver", "IteratedLocalSearchSolver"),
    ("solvers.multi_objective_genetic_solver", "MultiObjectiveGeneticSolver"),
    ("solvers.hybrid_genetic_solver", "HybridGeneticSolver"),
    ("solvers.dfj_solver", "DFJSolver"),
    ("solvers.mtz_solver", "MTZSolver"),
    ("solvers.branch_and_bound_solver", "BranchAndBoundSolver"),
    ("solvers.held_karp_solver", "HeldKarpSolver"),
    ("solvers.ant_colony_solver", "AntColonySolver"),
    ("solvers.particle_swarm_solver", "ParticleSwarmSolver"),
    # External/heavy
    ("solvers.lkh_solver", "LKHSolver"),
    ("solvers.concorde_solver", "ConcordeSolver"),
    ("solvers.concorde_hybrid_solver", "ConcordeHybridSolver"),
]


def try_import_solver(module_path: str, class_name: str) -> Optional[type]:
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls
    except Exception:
        return None


async def run_solver(solver_cls: type, problem: TestGraphV2Problem, per_solver_timeout: int) -> Tuple[str, bool, float, int]:
    name = solver_cls.__name__
    if name in SKIP_SOLVERS:
        return name, False, 0.0, 0
    try:
        solver = solver_cls(problem_types=[problem])
        start = time.time()
        route = await solver.solve_problem(problem, timeout=per_solver_timeout)
        elapsed = time.time() - start
        route_len = len(route) if route else 0
        return name, True, elapsed, route_len
    except Exception:
        return name, False, 0.0, 0


async def main():
    problems_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'problems')
    files = [os.path.join(problems_dir, f) for f in os.listdir(problems_dir) if f.endswith('.prob')]
    files.sort()
    if not files:
        print("No .prob files found in problems/")
        return

    # Test parameters
    per_solver_timeout = 10  # seconds per solver
    per_problem_limit = 150  # nodes per problem to cap runtime

    # Resolve available solver classes dynamically
    solver_classes: List[type] = []
    for module_path, class_name in SOLVER_SPECS:
        cls = try_import_solver(module_path, class_name)
        if cls is not None:
            solver_classes.append(cls)

    for path in files:
        try:
            coords = read_prob(path, limit=per_problem_limit)
            dist = build_euclidean_dist(coords)
            problem = make_problem(dist)
        except Exception as e:
            print(f"Problem {os.path.basename(path)}: failed to load ({e})")
            continue

        print(f"\n=== Problem: {os.path.basename(path)} | nodes={dist.shape[0]} ===")
        for solver_cls in solver_classes:
            name, ok, elapsed, route_len = await run_solver(solver_cls, problem, per_solver_timeout)
            status = "OK" if ok else "SKIP" if name in SKIP_SOLVERS else "ERR"
            print(f"{name:30s} {status:4s} time={elapsed:6.2f}s route_len={route_len}")


if __name__ == '__main__':
    asyncio.run(main())


