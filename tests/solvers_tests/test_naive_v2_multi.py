import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solvers import NaiveMultiSolver

PROBLEM_NUM = 10


class DummyMultiProblem:
    def __init__(self, n_nodes, n_salesmen):
        self.n_nodes = n_nodes
        self.n_salesmen = n_salesmen


def main():
    import asyncio
    solver = NaiveMultiSolver()

    os.makedirs('results', exist_ok=True)
    out_path = 'results/naive_v2_multi_results.csv'
    if os.path.exists(out_path):
        os.remove(out_path)

    for problem_id in range(PROBLEM_NUM):
        problem = DummyMultiProblem(n_nodes=10, n_salesmen=3)
        try:
            tours = asyncio.get_event_loop().run_until_complete(solver.solve(problem, future_id=0))
        except RuntimeError:
            import asyncio as aio
            loop = aio.new_event_loop()
            aio.set_event_loop(loop)
            tours = loop.run_until_complete(solver.solve(problem, future_id=0))
        # Write the number of tours produced
        with open(out_path, 'a') as f:
            f.write(f"{problem_id},{len(tours)}\n")
        print(f'Problem {problem_id} produced {len(tours)} tours')


if __name__ == '__main__':
    main()


