import os
import sys
import types
import time
import asyncio
from typing import List, Tuple

import numpy as np


def ensure_dummy_solvers_package(project_root: str) -> None:
    solvers_dir = os.path.join(project_root, 'solvers')
    if solvers_dir not in sys.path:
        sys.path.insert(0, solvers_dir)
    # Install a dummy package for 'solvers' to avoid executing package __init__
    if 'solvers' not in sys.modules:
        pkg = types.ModuleType('solvers')
        pkg.__path__ = [solvers_dir]  # type: ignore[attr-defined]
        sys.modules['solvers'] = pkg


def read_prob(path: str) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    coords: List[Tuple[float, float]] = []
    for line in lines[1:]:
        try:
            a, b = line.split()
            coords.append((float(a), float(b)))
        except Exception:
            continue
    if len(coords) < 2:
        raise ValueError(f"Not enough coordinates in {path}")
    return np.array(coords, dtype=float)


def build_euclidean_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, 0.0)
    return dist


class TestGraphV2Problem:
    def __init__(self, edges: np.ndarray):
        self.problem_type = "Metric TSP"
        self.n_nodes = edges.shape[0]
        self.edges = edges


async def run_solver(solver_cls, problem: TestGraphV2Problem, timeout_seconds: int = 10):
    solver = solver_cls(problem_types=[problem])
    return await solver.solve_problem(problem, timeout=timeout_seconds)


