import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tools.search import search_messages  # noqa: E402


def show(result):
    if "error" in result:
        print("  ERROR:", result["error"])
        return
    for r in result["results"]:
        print(f"  [{r['similarity']}] {r['record_id']} | {r['receiver_department']} | {r['subcategory']}")
        print(f"      {r['message'][:150]}...")


print("1. Messages about mentoring and coaching")
show(search_messages("mentoring and coaching new team members", limit=3))

print("\n2. Same search, but only messages received by Engineering")
show(search_messages("mentoring and coaching new team members", department="Engineering", limit=3))

print("\n3. Handling an outage, only in Operational Excellence")
show(search_messages("fixing an outage or urgent incident",
                     category="Operational Excellence & Delivery", limit=3))

print("\n4. A department that doesn't exist")
show(search_messages("teamwork", department="Space Program"))