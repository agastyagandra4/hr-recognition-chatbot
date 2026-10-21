import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from core.tools.search import embed, get_collection  # noqa: E402

# Load the cleaned messages from the database
with sqlite3.connect("data/recognition.db") as conn:
    df = pd.read_sql("SELECT * FROM recognitions", conn)

print(f"Embedding {len(df)} messages (the first run downloads the model, about 90 MB)...")
embeddings = embed(df["message"].tolist())

metadata_columns = [
    "giver_department",
    "receiver_department",
    "giver_title",
    "receiver_title",
    "category_name",
    "subcategory_name",
    "award_title",
]

collection = get_collection()
collection.upsert(
    ids=df["record_id"].tolist(),
    documents=df["message"].tolist(),
    embeddings=embeddings,
    metadatas=df[metadata_columns].to_dict("records"),
)

print(f"Search index now holds {collection.count()} messages")