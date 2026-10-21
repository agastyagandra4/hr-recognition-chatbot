import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tools.stats import (  # noqa: E402
    category_breakdown,
    dataset_overview,
    department_flows,
    recognitions_by_department,
)

print("1. Dataset overview")
print(dataset_overview())

print("\n2. Which departments receive the most recognition? (top 5)")
for row in recognitions_by_department("received")["results"][:5]:
    print(row)

print("\n3. What is Finance recognized for? (top 3)")
finance = category_breakdown("Finance")
print("Total:", finance["total"])
for row in finance["results"][:3]:
    print(row)

print("\n4. Strongest cross-department flows (top 5)")
for row in department_flows(limit=5)["results"]:
    print(row)

print("\n5. A department that doesn't exist")
print(category_breakdown("Space Program"))