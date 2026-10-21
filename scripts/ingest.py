import json
import re
import sqlite3

import pandas as pd
import yaml

# ---------- Step 2: Load and rename ----------

with open("config/schema.yaml") as f:
    schema = yaml.safe_load(f)

df = pd.read_csv("data/raw/annotated_results.csv")

rename_map = {csv_name: our_name for our_name, csv_name in schema["columns"].items()}
df = df.rename(columns=rename_map)

# ---------- Step 3: Clean the data ----------

for col in df.columns:
    df[col] = df[col].str.strip()

df["giver_title"] = df["giver_title"].fillna("Unknown")
df["award_title"] = df["award_title"].fillna("")
df.insert(0, "record_id", [f"R{i:04d}" for i in range(1, len(df) + 1)])

print(f"Loaded and cleaned {len(df)} records")

# ---------- Step 4: Check the category labels ----------

with open("data/raw/cleaned_taxonomy_2.json") as f:
    taxonomy_json = json.load(f)["recognition_taxonomy"]

rows = []
for category_name, category in taxonomy_json.items():
    for sub in category["subcategories"]:
        rows.append({
            "category_id": category["id"],
            "category_name": category_name,
            "subcategory_id": sub["id"],
            "subcategory_name": sub["name"],
        })
taxonomy = pd.DataFrame(rows)

unknown = set(df["subcategory_id"]) - set(taxonomy["subcategory_id"])
print(f"Invalid labels: {sorted(unknown) if unknown else 'none'}")

# ---------- Step 5: Turn job titles into departments ----------

with open("config/departments.yaml") as f:
    dept_config = yaml.safe_load(f)

overrides = {title.lower(): dept for title, dept in dept_config["overrides"].items()}


def find_department(title):
    """Return the department for one job title."""
    clean = title.lower()
    if clean in overrides:
        return overrides[clean]
    for rule in dept_config["rules"]:
        for keyword in rule["keywords"]:
            if re.search(r"\b" + re.escape(keyword), clean):
                return rule["department"]
    return "Other"


all_titles = pd.concat([df["giver_title"], df["receiver_title"]]).unique()
title_to_dept = {title: find_department(title) for title in all_titles}

df["giver_department"] = df["giver_title"].map(title_to_dept)
df["receiver_department"] = df["receiver_title"].map(title_to_dept)

other_count = sum(1 for dept in title_to_dept.values() if dept == "Other")
print(f"Departments assigned: {len(title_to_dept) - other_count} of {len(title_to_dept)} titles")

# ---------- Step 6: Save everything into a database ----------

# Build the title table: each title, its department, and how often it appears
title_counts = pd.concat([df["giver_title"], df["receiver_title"]]).value_counts()
title_table = pd.DataFrame({
    "title": title_counts.index,
    "department": [title_to_dept[t] for t in title_counts.index],
    "records": title_counts.values,
})

# Save the three tables into one SQLite file
with sqlite3.connect("data/recognition.db") as conn:
    df.to_sql("recognitions", conn, if_exists="replace", index=False)
    taxonomy.to_sql("taxonomy", conn, if_exists="replace", index=False)
    title_table.to_sql("title_departments", conn, if_exists="replace", index=False)

    # Read it back with a SQL query to prove the save worked
    check = pd.read_sql(
        """
        SELECT receiver_department, COUNT(*) AS recognitions
        FROM recognitions
        GROUP BY receiver_department
        ORDER BY recognitions DESC
        LIMIT 5
        """,
        conn,
    )

# Also save the title table as a CSV you can open and review
title_table.to_csv("data/department_review.csv", index=False)

print("\nSaved data/recognition.db and data/department_review.csv")
print("\nTop 5 departments, read back from the database with SQL:")
print(check)