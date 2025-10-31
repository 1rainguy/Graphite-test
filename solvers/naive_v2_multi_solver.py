from typing import List
import random

def split_into_sublists(lst, m):
    random.shuffle(lst)
    k, r = divmod(len(lst), m)
    sublists = [lst[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(m)]
    return sublists

class NaiveMultiSolver:
    '''
    Mock solver for comparison. Returns the route as per the random selection.
    '''
    def __init__(self):
        pass

    def solve(self, formatted_problem) -> List[List[int]]:
        m = formatted_problem.n_salesmen
        n = formatted_problem.n_nodes
        city_paths = split_into_sublists(list(range(1,n)), m)
        completed_tours = []
        for path in city_paths:
            completed_tours.append([0] + path + [0])
        return completed_tours
