from typing import List
from .common_utils import nearest_neighbor, two_opt_improve
from scipy.spatial import distance
import numpy as np
import time
import os

class HPNSolver:
    '''
    HPN solver with fallback to nearest neighbor + 2-opt when model unavailable
    '''
    def __init__(self, time_limit: int = 100, weights_fp: str = None):
        self.time_limit = time_limit
        self.weights_fp = weights_fp
        self.has_model = False
        
        try:
            import torch
            from graphite.models.hybrid_pointer_network import HPN
            
            if weights_fp and os.path.exists(weights_fp):
                self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                self.critic = HPN(n_feature=2, n_hidden=128)
                self.critic = self.critic.to(self.device)
                self.critic.eval()
                checkpoint = torch.load(weights_fp, map_location=self.device)
                self.critic.load_state_dict(checkpoint['model_baseline'])
                self.has_model = True
        except:
            self.has_model = False

    def solve(self, formatted_problem, post_process: bool = False) -> List[int]:
        coordinates = formatted_problem
        size = len(coordinates)
        start_time = time.time()
        
        if size <= 2:
            return list(range(size)) + [0]
        
        if not self.has_model:
            dmatrix = distance.cdist(coordinates, coordinates, 'euclidean')
            tour = nearest_neighbor(dmatrix, start=0, start_time=start_time, hard_limit=self.time_limit)
            if post_process:
                tour = two_opt_improve(tour, dmatrix, start_time, self.time_limit)
            if tour and len(tour) == size:
                return tour + [tour[0]]
            return list(range(size)) + [0]
        
        try:
            import torch
            B = 1
            X = torch.from_numpy(coordinates).float().to(self.device)
            X = X.unsqueeze(0)
            mask = torch.zeros(B, size).to(self.device)
            
            solution = []
            Y = X.view(B, size, 2)
            x = Y[:, 0, :]
            h = None
            c = None
            Transcontext = None
            visited_indices = []
            
            for k in range(size):
                if (time.time() - start_time) >= self.time_limit:
                    break
                Transcontext, output, h, c, _ = self.critic(Transcontext, x=x, X_all=X, h=h, c=c, mask=mask)
                idx = torch.argmax(output, dim=1)
                x = Y[[i for i in range(B)], idx.data]
                solution.append(x.cpu().numpy())
                visited_indices.append(idx.cpu().numpy())
                mask[[i for i in range(B)], idx.data] += -np.inf
            
            solution.append(solution[0])
            visited_indices.append(visited_indices[0])
            graph = np.array(solution)
            route = list(range(size)) + [0]
            
            if post_process:
                best_indices = self.post_process(B, route, graph, size, visited_indices)
                tour_1d = [int(arr[0]) for arr in best_indices]
            else:
                tour_1d = [int(arr[0]) for arr in visited_indices]
            
            start_index = tour_1d.index(0) if 0 in tour_1d else 0
            min_path = tour_1d[start_index:] + tour_1d[:start_index] + [0]
            return min_path
        except Exception as e:
            dmatrix = distance.cdist(coordinates, coordinates, 'euclidean')
            tour = nearest_neighbor(dmatrix, start=0, start_time=start_time, hard_limit=self.time_limit)
            if post_process:
                tour = two_opt_improve(tour, dmatrix, start_time, self.time_limit)
            if tour and len(tour) == size:
                return tour + [tour[0]]
            return list(range(size)) + [0]

    def post_process(self, B, route, graph, size, visited_indices):
        best_solutions = []
        for b in range(B):
            best = route.copy()
            graph_ = graph[:, b, :].copy()
            dmatrix = distance.cdist(graph_, graph_, 'euclidean')
            improved = True
            
            while improved:
                improved = False
                for i in range(size):
                    for j in range(i + 2, size + 1):
                        old_dist = dmatrix[best[i], best[i + 1]] + dmatrix[best[j], best[j - 1]]
                        new_dist = dmatrix[best[j], best[i + 1]] + dmatrix[best[i], best[j - 1]]
                        if new_dist < old_dist:
                            best[i + 1:j] = best[j - 1:i:-1]
                            improved = True
            
            best_solutions.append(best)
        
        best_indices = [visited_indices[idx] for idx in best_solutions[0]]
        return best_indices
