import sqlite3

import pandas as pd

DB_PATH = "data/recognition.db"


def run_query(sql, params=()):
    """Run a SQL query on the database and return the result as a table."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn, params=params)


def dataset_overview():
    """Basic facts about the whole dataset."""
    totals = run_query(
        """
        SELECT COUNT(*) AS records,
               COUNT(DISTINCT receiver_department) AS departments,
               COUNT(DISTINCT category_id) AS categories,
               COUNT(DISTINCT subcategory_id) AS subcategories_used
        FROM recognitions
        """
    )
    unused = run_query(
        """
        SELECT subcategory_id, subcategory_name
        FROM taxonomy
        WHERE subcategory_id NOT IN (SELECT DISTINCT subcategory_id FROM recognitions)
        """
    )
    result = totals.iloc[0].to_dict()
    result = {key: int(value) for key, value in result.items()}
    result["subcategories_with_no_records"] = unused.to_dict("records")
    return result


def recognitions_by_department(direction="received"):
    """How many recognitions each department received (or gave)."""
    columns = {"received": "receiver_department", "given": "giver_department"}
    if direction not in columns:
        return {"error": "direction must be 'received' or 'given'"}
    column = columns[direction]

    table = run_query(
        f"""
        SELECT {column} AS department, COUNT(*) AS recognitions
        FROM recognitions
        GROUP BY {column}
        ORDER BY recognitions DESC
        """
    )
    table["share"] = (table["recognitions"] / table["recognitions"].sum()).round(3)
    return {"direction": direction, "results": table.to_dict("records")}


def category_breakdown(department=None):
    """Recognitions per category and subcategory, optionally for one receiving department.
    Also lists subcategories with zero records, so 'none' answers are visible."""
    join_condition = "r.subcategory_id = t.subcategory_id"
    params = ()
    if department:
        join_condition += " AND r.receiver_department = ?"
        params = (department,)

    table = run_query(
        f"""
        SELECT t.category_name, t.subcategory_name, COUNT(r.record_id) AS recognitions
        FROM taxonomy t
        LEFT JOIN recognitions r ON {join_condition}
        GROUP BY t.category_name, t.subcategory_name
        ORDER BY recognitions DESC
        """,
        params,
    )

    total = int(table["recognitions"].sum())
    if total == 0:
        return {"error": f"No records found for department '{department}'"}

    return {
        "department": department or "all",
        "total": total,
        "results": table[table["recognitions"] > 0].to_dict("records"),
        "subcategories_with_no_records": table[table["recognitions"] == 0]["subcategory_name"].tolist(),
    }
def department_flows(limit=10):
    """The strongest giver-department -> receiver-department pairs, across departments only."""
    table = run_query(
        """
        SELECT giver_department, receiver_department, COUNT(*) AS recognitions
        FROM recognitions
        WHERE giver_department != receiver_department
        GROUP BY giver_department, receiver_department
        ORDER BY recognitions DESC
        LIMIT ?
        """,
        (limit,),
    )
    return {"results": table.to_dict("records")}