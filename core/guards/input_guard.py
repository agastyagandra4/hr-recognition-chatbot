import re
from pathlib import Path

import yaml
from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "config" / "guardrails.yaml") as f:
    GUARD_CONFIG = yaml.safe_load(f)

# A small, fast model is enough for labeling
classifier = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

CLASSIFIER_PROMPT = """You label questions sent to an HR recognition analytics chatbot.
The chatbot answers questions about recognition messages between employees: counts by
department or role, categories of recognition, example messages, and how teams are connected.

Labels:
- in_scope: about recognition patterns, departments, roles, categories, messages, the dataset,
  or what the chatbot can do. Short follow-ups like "what about Marketing?" are in_scope.
  Comparing departments or roles (including which gets the least) is in_scope.
- off_topic: unrelated to recognition data (weather, coding help, general knowledge, poems).
- individual_judgment: asks to judge, rank, or evaluate a specific person's performance or worth.
- protected_attribute: asks about gender, race, age, religion, disability, nationality,
  sexual orientation, or similar characteristics.
- workforce_decision: asks who to promote, fire, lay off, pay more, or how to make
  other employment decisions.

Reply with exactly one label and nothing else."""


def classify(question):
    """Return (label, note). If anything goes wrong, let the question through (fail-open)."""
    try:
        response = classifier.invoke([("system", CLASSIFIER_PROMPT), ("human", question)])
        label = re.sub(r"[^a-z_]", "", response.content.strip().lower())
    except Exception as error:
        return "in_scope", f"classifier error, let through: {error}"

    if label not in GUARD_CONFIG["labels"]:
        return "in_scope", f"unknown label '{label}', let through"
    return label, "classified"