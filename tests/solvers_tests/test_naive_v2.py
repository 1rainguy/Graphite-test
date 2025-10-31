import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import csv
import numpy as np
from solvers.naive_v2_solver import NaiveSolver
from tests.utils import read_prob

def test_naive_v2_solver():
    solver = NaiveSolver()
    results = []
    
    problems_dir = os.path.join(PROJECT_ROOT, 'problems')
    test_problems = [f'{i}.prob' for i in range(10)]
    
    for prob_file in test_problems:
        prob_path = os.path.join(problems_dir, prob_file)
        if not os.path.exists(prob_path):
            continue
            
        try:
            coords = read_prob(prob_path)
            n = len(coords)
            if n < 2:
                continue
            
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist[i][j] = np.linalg.norm(coords[i] - coords[j])
            
            route = list(range(n)) + [0]
            solution = solver.solve(route)
            
            if solution and len(solution) == n + 1:
                tour_length = 0
                for i in range(n):
                    tour_length += dist[solution[i]][solution[i + 1]]
                
                results.append({
                    'problem': prob_file,
                    'n_nodes': n,
                    'tour_length': tour_length,
                    'status': 'success'
                })
            else:
                results.append({
                    'problem': prob_file,
                    'n_nodes': n,
                    'tour_length': None,
                    'status': 'invalid_solution'
                })
        except Exception as e:
            results.append({
                'problem': prob_file,
                'n_nodes': None,
                'tour_length': None,
                'status': f'error: {str(e)}'
            })
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
    results_file = os.path.join(PROJECT_ROOT, 'results', 'naive_v2_results.csv')
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['problem', 'n_nodes', 'tour_length', 'status'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {results_file}")

if __name__ == '__main__':
    test_naive_v2_solver()
