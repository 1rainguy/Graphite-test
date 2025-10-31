from typing import Any


def valid_problem(problem: Any) -> bool:
    # Minimal validation placeholder
    return hasattr(problem, "problem_type")


def timeout(seconds: int):  # no-op decorator for compatibility
    def decorator(func):
        return func
    return decorator

# Additional placeholders used by some solvers
def get_multi_minmax_tour_distance(*args, **kwargs):
    return 0.0, 0.0

def get_portfolio_distribution_similarity(*args, **kwargs):
    return 0.0

def normalize_coordinates(coords):
    return coords


