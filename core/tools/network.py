import sqlite3

import networkx as nx
import pandas as pd

DB_PATH = "data/recognition.db"

# Which database columns to use for each level of detail
LEVELS = {
    "department": ("giver_department", "receiver_department"),
    "role": ("giver_title", "receiver_title"),
}

_graphs = {}


def build_graph(level="department"):
    """Directed graph: an arrow from giver to receiver, weighted by number of recognitions."""
    if level in _graphs:
        return _graphs[level]

    giver_col, receiver_col = LEVELS[level]
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT {giver_col} AS giver, {receiver_col} AS receiver FROM recognitions", conn)

    # Keep only recognitions between different groups, and drop placeholder values
    placeholders = ["Unknown", "Other"]
    df = df[(df["giver"] != df["receiver"])
            & ~df["giver"].isin(placeholders)
            & ~df["receiver"].isin(placeholders)]
    counts = df.groupby(["giver", "receiver"]).size().reset_index(name="weight")

    graph = nx.DiGraph()
    for row in counts.itertuples():
        graph.add_edge(row.giver, row.receiver, weight=int(row.weight))

    _graphs[level] = graph
    return graph


def combined_graph(level):
    """Undirected version: A->B and B->A are added together into one connection."""
    combined = nx.Graph()
    for a, b, data in build_graph(level).edges(data=True):
        if combined.has_edge(a, b):
            combined[a][b]["weight"] += data["weight"]
        else:
            combined.add_edge(a, b, weight=data["weight"])

    # More recognition = closer together, so distance is 1 / weight
    for a, b, data in combined.edges(data=True):
        data["distance"] = 1 / data["weight"]
    return combined


def mutual_pairs(level="department", limit=5):
    """Pairs that recognize each other in BOTH directions, strongest first."""
    if level not in LEVELS:
        return {"error": "level must be 'department' or 'role'"}
    graph = build_graph(level)

    pairs = []
    for a, b in graph.edges():
        if a < b and graph.has_edge(b, a):
            a_to_b = graph[a][b]["weight"]
            b_to_a = graph[b][a]["weight"]
            pairs.append({"pair": [a, b], "a_to_b": a_to_b, "b_to_a": b_to_a,
                          "weaker_direction": min(a_to_b, b_to_a)})

    pairs.sort(key=lambda p: p["weaker_direction"], reverse=True)
    return {"level": level, "results": pairs[:limit]}


def most_connected(level="department", limit=5):
    """Who holds the organization together: number of partners, plus a bridge score."""
    if level not in LEVELS:
        return {"error": "level must be 'department' or 'role'"}
    graph = combined_graph(level)

    bridge_scores = nx.betweenness_centrality(graph, weight="distance")

    results = []
    for node in graph.nodes():
        results.append({
            "name": node,
            "partners": graph.degree(node),
            "bridge_score": round(bridge_scores[node], 3),
        })

    results.sort(key=lambda r: r["bridge_score"], reverse=True)
    return {"level": level, "results": results[:limit]}


def connection_path(source, target, level="department"):
    """The strongest chain of recognition linking two groups."""
    if level not in LEVELS:
        return {"error": "level must be 'department' or 'role'"}
    graph = combined_graph(level)

    for name in (source, target):
        if name not in graph:
            return {"error": f"'{name}' was not found at the {level} level"}

    try:
        path = nx.shortest_path(graph, source, target, weight="distance")
    except nx.NetworkXNoPath:
        return {"error": f"No recognition path connects {source} and {target}"}

    return {"level": level, "from": source, "to": target, "path": path, "steps": len(path) - 1}