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

class NearestNeighbourMultiSolver4:
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
    
    def solve(self, formatted_problem)->List[int]:
        def subtour_distance(distance_matrix, subtour):
            subtour = np.array(subtour)
            next_points = np.roll(subtour, -1) 
            distances = distance_matrix[subtour, next_points] 
            total_distance = np.sum(distances)
            return total_distance
        
        # construct m tours
        m = formatted_problem.n_salesmen
        distance_matrix = np.array(formatted_problem.edges)
        unvisited = [city for city in range(len(distance_matrix)) if city not in set(formatted_problem.depots)]
        tours = self.get_starting_tours(formatted_problem.depots, distance_matrix)

        for _, first_city in tours:
            unvisited.remove(first_city)

        distances = [subtour_distance(distance_matrix, subtour) for subtour in tours]
        constraints = formatted_problem.constraint # List of constraints per salesman
        demands = formatted_problem.demand # List of demand per node

        tours_demand = [[demands[j] for j in i] for i in tours]

        while unvisited:
            # print("TOUR LENS", [len(tour) for tour in tours])
            chosen_index = distances.index(min(distances))
            chosen_subtour = tours[chosen_index]

            min_distance = np.inf
            chosen_city = None
            for city in unvisited:
                
                new_distance = distances[chosen_index] - distance_matrix[chosen_subtour[-1]][0] + distance_matrix[chosen_subtour[-1]][city] + distance_matrix[city][0]
                if new_distance < min_distance and sum(tours_demand[chosen_index]) + demands[city] <= constraints[chosen_index]:
                    chosen_city = city
                    min_distance = new_distance
            if chosen_city is not None and chosen_city in unvisited:
                distances[chosen_index] = min_distance
                tours[chosen_index] = chosen_subtour + [chosen_city]
                tours_demand[chosen_index].append(demands[chosen_city])
                unvisited.remove(chosen_city)

            # Ensure we do not exceed constraint
            if sum(tours_demand[chosen_index]) >= constraints[chosen_index]:  # Exclude depot
                distances[chosen_index] = np.inf

        print("TOUR DEMAND MET", [sum(tour_de) for tour_de in tours_demand])
        return [tour + [depot] for tour, depot in zip(tours, formatted_problem.depots.copy())] # complete each subtour back to source depot
    
    
    