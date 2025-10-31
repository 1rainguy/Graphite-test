from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GraphV1Problem:
    n_nodes: int = 0
    directed: bool = False
    problem_type: str = "General TSP"
    edges: Any = None


@dataclass
class GraphV2Problem:
    problem_type: str = "Metric TSP"
    n_nodes: int = 0
    selected_ids: Optional[List[int]] = None
    cost_function: str = "Geom"
    dataset_ref: Optional[str] = None
    edges: Any = None


@dataclass
class GraphV2ProblemMulti:
    problem_type: str = "Multi TSP"
    n_nodes: int = 0
    selected_ids: Optional[List[int]] = None
    dataset_ref: Optional[str] = None
    n_salesmen: int = 1
    depots: Optional[List[int]] = None
    single_depot: bool = True
    edges: Any = None


@dataclass
class GraphV2ProblemMultiConstrained(GraphV2ProblemMulti):
    time_windows: Any = None
    capacities: Any = None


@dataclass
class GraphV2ProblemMultiConstrainedTW(GraphV2ProblemMulti):
    time_windows: Any = None


@dataclass
class GraphV1PortfolioProblem:
    n_nodes: int = 0
    problem_type: str = "Portfolio TSP"
    edges: Any = None


# Simple placeholder used by some files
@dataclass
class GraphV2Synapse:
    pass


