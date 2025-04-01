#This is a file dedicated to specific sampling functions.
#For example, with the help of sample_scatter_with_coefficent, we can sample from the distribution in a way 
# such that we space out our rocks
# this reduces the number of logs needed to cover the terrain

import random
import math

def sample_scatter_with_coefficient(scatter_range, previous_positions, spacing_coefficient, max_attempts=1000):
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    min_x, max_x = scatter_range
    min_y, max_y = scatter_range  # assuming square region

    max_distance = math.sqrt(2) * (max_x - min_x) / 2
    min_distance = spacing_coefficient * max_distance

    for _ in range(max_attempts):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        new_point = (x, y)

        if all(euclidean_distance(new_point, p) >= min_distance for p in previous_positions):
            return new_point

    # If no suitable point is found, return None or relax constraints
    return None