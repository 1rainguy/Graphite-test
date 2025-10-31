import math

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def tour_length(tour, points):
    return sum(euclidean(points[tour[i]], points[tour[i+1]]) for i in range(len(tour)-1))

def two_opt(points, tour):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                # current edges: (i-1,i) and (j,j+1)
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[j+1]
                delta = (euclidean(points[a], points[c]) + euclidean(points[b], points[d])
                         - euclidean(points[a], points[b]) - euclidean(points[c], points[d]))
                if delta < -1e-9:
                    # perform 2-opt swap: reverse segment [i..j]
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
        if not improved:
            break
    return tour
