import json
import sqlite3
from typing import Literal, Optional

from langchain_core.tools import tool

from core.tools import network, search, stats


def to_json(result):
    """Tools return text to the AI, so convert each result dictionary into JSON text."""
    return json.dumps(result, default=str)


# ---------- Statistics tools ----------

@tool
def get_dataset_overview() -> str:
    """Basic facts about the dataset: number of records, departments, categories,
    and which subcategories have no records at all. Use for 'how many' or 'are there any' questions."""
    return to_json(stats.dataset_overview())


@tool
def get_recognitions_by_department(direction: Literal["received", "given"] = "received") -> str:
    """Count recognitions per department, with each department's share of the total.
    direction='received' counts what each department received; 'given' counts what it gave."""
    return to_json(stats.recognitions_by_department(direction))


@tool
def get_category_breakdown(department: Optional[str] = None) -> str:
    """Count recognitions per category and subcategory, i.e. WHAT people are recognized for.
    Optionally limit to recognitions received by one department (use an exact department name)."""
    return to_json(stats.category_breakdown(department))


@tool
def get_department_flows(limit: int = 10) -> str:
    """The strongest one-way recognition flows between different departments
    (giver department -> receiver department), largest first."""
    return to_json(stats.department_flows(limit))


# ---------- Search tool ----------

@tool
def search_recognition_messages(
    query: str,
    department: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Find actual recognition messages by meaning (semantic search). Use when the user wants
    to see examples, quotes, or what messages say about a topic. Optional filters: the receiving
    department and the category name (use exact names)."""
    return to_json(search.search_messages(query, department, category, limit))


# ---------- Network tools ----------

@tool
def find_mutual_pairs(level: Literal["department", "role"] = "department", limit: int = 5) -> str:
    """Pairs of departments (or job roles) that recognize each other in BOTH directions,
    strongest two-way relationships first."""
    return to_json(network.mutual_pairs(level, limit))


@tool
def find_most_connected(level: Literal["department", "role"] = "department", limit: int = 5) -> str:
    """Which departments (or job roles) connect the organization: number of partners and a
    bridge score (betweenness centrality). Use for hubs, bridges, or 'who connects teams' questions."""
    return to_json(network.most_connected(level, limit))


@tool
def find_connection_path(
    source: str, target: str, level: Literal["department", "role"] = "department"
) -> str:
    """The strongest chain of recognition linking two departments (or two job roles)."""
    return to_json(network.connection_path(source, target, level))


ALL_TOOLS = [
    get_dataset_overview,
    get_recognitions_by_department,
    get_category_breakdown,
    get_department_flows,
    search_recognition_messages,
    find_mutual_pairs,
    find_most_connected,
    find_connection_path,
]


def system_prompt():
    """Instructions for the AI, including the exact department and category names."""
    with sqlite3.connect(stats.DB_PATH) as conn:
        departments = [row[0] for row in conn.execute(
            "SELECT DISTINCT receiver_department FROM recognitions WHERE receiver_department != 'Other' ORDER BY 1")]
        categories = [row[0] for row in conn.execute(
            "SELECT DISTINCT category_name FROM taxonomy ORDER BY 1")]

    return f"""You are an HR analytics assistant that answers questions about an organization's
employee recognition data (1,000 recognition messages between employees).

Rules:
- Always use the tools to get facts. Never guess or invent numbers, names, or messages.
- Every number in your answer must come from a tool result.
- If a question needs more than one tool, call them one after another.
- Use these exact department names in tool inputs: {", ".join(departments)}.
- Use these exact category names in tool inputs: {", ".join(categories)}.
- The data has no dates, so you cannot answer questions about trends over time. Say so.
- Match the user's topic to the subcategory with that meaning. Never substitute a different
  subcategory that only sounds similar. If the matching subcategory has 0 records, say there are none.
- If a question is unrelated to recognition data, politely say what you can help with instead.
- Keep answers short and clear. Use plain language."""