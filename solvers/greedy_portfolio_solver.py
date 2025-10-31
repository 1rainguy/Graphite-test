# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
import numpy as np
import time
import asyncio
import random
import math
from graphite.base.subnetPool import SubnetPool
from copy import deepcopy

class GreedyPortfolioSolver:
    '''
    This solver is a constructive nearest_neighbour algorithm that assigns cities to subtours based on the min increase in objective function value.
    '''
    def __init__(self):
        pass
    
    def get_valid_start(self, depot_id, distance_matrix, taken_nodes:list[int]=[], selection_range:int=5) -> int:
        distances = [(city_id, distance) for city_id, distance in enumerate(distance_matrix[depot_id].copy())]
        # reverse sort the copied list and pop from it
        assert (selection_range + len(taken_nodes)) < len(distances)
        distances.sort(reverse=True, key=lambda x: x[1])
        closest_cities = []
        while len(closest_cities) < selection_range:
            selected_city = None
            while not selected_city:
                city_id, distance = distances.pop()
                if city_id not in taken_nodes:
                    selected_city = city_id
            closest_cities.append(selected_city)
        return closest_cities[0]
        
    def get_starting_tours(self, depots, distance_matrix):
        taken_nodes = depots.copy()
        initial_incomplete_tours = []
        for depot in depots:
            first_visit = self.get_valid_start(depot, distance_matrix, taken_nodes)
            initial_incomplete_tours.append([depot, first_visit])
            taken_nodes.append(first_visit)
        return initial_incomplete_tours
    
    def solve(self, formatted_problem):
        """
        formatted_problem:
            problem_type: Literal['PortfolioReallocation'] = Field('PortfolioReallocation', description="Problem Type")
            n_portfolio: int = Field(3, description="Number of Portfolios")
            initialPortfolios: List[List[Union[float, int]]] = Field([[0]*100]*3, description="Number of tokens in each subnet for each of the n_portfolio eg. 3 portfolios with 0 tokens in any of the 100 subnets")
            constraintValues: List[Union[float, int]] = Field([1.0]*100, description="Overall Percentage for each subnet in equivalent TAO after taking the sum of all portfolios; they do not need to add up to 100%")
            constraintTypes: List[str] = Field(["ge"]*100, description="eq = equal to, ge = greater or equals to, le = lesser or equals to - the value provided in constraintValues")
            pools: List[Union[float, int]] = Field([[1.0, 1.0]]*100, description="Snapshot of current pool states of all subnets when problem is issued, list idx = netuid, [num_tao_tokens, num_alpha_tokens]")
        
        output:
            bool or solution
            solution = [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_tokens], ... ]
        """

        ### Individual portfolio level swaps required
        def instantiate_pools(pools):
            current_pools: List[SubnetPool] = []
            for netuid, pool in enumerate(pools):
                current_pools.append(SubnetPool(pool[0], pool[1], netuid))
            return current_pools

        start_pools = instantiate_pools(formatted_problem.pools)

        initialPortfolios = deepcopy(formatted_problem.initialPortfolios)
        total_tao = 0
        portfolio_tao = [0] * formatted_problem.n_portfolio
        portfolio_swaps = [] # [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_tokens], ... ]
        for idx, portfolio in enumerate(initialPortfolios):
            for netuid, alpha_token in enumerate(portfolio):
                if alpha_token > 0:
                    emitted_tao = start_pools[netuid].swap_alpha_to_tao(alpha_token)
                    portfolio_swaps.append([idx, netuid, 0, int(alpha_token)])
                    total_tao += emitted_tao
                    portfolio_tao[idx] += emitted_tao

        for netuid, constraint_type in enumerate(formatted_problem.constraintTypes):
            if netuid != 0:
                constraint_value = formatted_problem.constraintValues[netuid]
                tao_required = constraint_value/100 * total_tao
                if constraint_type == "eq" or constraint_type == "ge":
                    for idx in range(len(portfolio_tao)):
                        tao_to_swap = min(portfolio_tao[idx], tao_required)
                        if tao_to_swap > 0:
                            alpha_emitted = start_pools[netuid].swap_tao_to_alpha(tao_to_swap)
                            portfolio_swaps.append([idx, 0, netuid, int(tao_to_swap)])
                            tao_required -= tao_to_swap
                            portfolio_tao[idx] -= tao_to_swap

        return portfolio_swaps
    
    
    