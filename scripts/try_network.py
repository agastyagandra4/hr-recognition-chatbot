import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tools.network import build_graph, connection_path, most_connected, mutual_pairs  # noqa: E402

graph = build_graph("department")
print(f"Department graph: {graph.number_of_nodes()} departments, {graph.number_of_edges()} connections")

print("\n1. Departments that recognize each other both ways (top 3)")
for row in mutual_pairs("department", limit=3)["results"]:
    print(row)

print("\n2. Departments that best connect the organization (top 5)")
for row in most_connected("department", limit=5)["results"]:
    print(row)

print("\n3. Roles that act as bridges between teams (top 5)")
for row in most_connected("role", limit=5)["results"]:
    print(row)

print("\n4. How is Legal connected to Linguistics?")
print(connection_path("Legal", "Linguistics"))

print("\n5. A department that doesn't exist")
print(connection_path("Legal", "Space Program"))