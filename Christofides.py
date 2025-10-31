import networkx as nx

def christofides_tsp(G):
    # Step 1: MST
    T = nx.minimum_spanning_tree(G)

    # Step 2: vertices of odd degree
    odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]

    # Step 3: Minimum weight perfect matching among odd-degree nodes
    import itertools
    import copy
    subgraph = G.subgraph(odd_degree_nodes)
    # Complete graph with weights
    complete = nx.Graph()
    for u, v in itertools.combinations(odd_degree_nodes, 2):
        complete.add_edge(u, v, weight=subgraph[u][v]['weight'] if subgraph.has_edge(u,v) else G[u][v]['weight'])
    matching = nx.algorithms.matching.min_weight_matching(complete, maxcardinality=True)

    # Combine MST and matching edges
    multi_graph = nx.MultiGraph()
    multi_graph.add_edges_from(T.edges(data=True))
    for u, v in matching:
        w = G[u][v]['weight']
        multi_graph.add_edge(u, v, weight=w)

    # Step 4: Eulerian tour
    euler_circuit = list(nx.eulerian_circuit(multi_graph))

    # Step 5: Shortcut to Hamiltonian tour
    visited = set()
    tour = []
    for u, v in euler_circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)
        if v not in visited:
            tour.append(v)
            visited.add(v)
    tour.append(tour[0])  # return to start
    return tour


