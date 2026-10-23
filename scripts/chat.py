import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from core.agent.graph import build_agent  # noqa: E402

agent = build_agent()


def new_config():
    """A fresh conversation ID, plus a limit on how many steps one question may take."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 12}


config = new_config()
print("HR Recognition Chatbot. Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in {"quit", "exit"}:
        break
    if not question:
        continue

    try:
        result = agent.invoke({"messages": [HumanMessage(content=question)]}, config)
    except GraphRecursionError:
        print("Bot: I couldn't finish that within the step limit. Try asking more specifically.\n")
        config = new_config()  # start a clean conversation after a failed run
        continue

    # Find which tools the AI used for this question
    new_messages = []
    for message in reversed(result["messages"]):
        if isinstance(message, HumanMessage):
            break
        new_messages.append(message)
    tools_used = [call["name"] for m in reversed(new_messages)
                  if isinstance(m, AIMessage) for call in m.tool_calls]
    label = result.get("guard_label", "in_scope")
    if label != "in_scope":
        print(f"  (guardrail: {label})")
    if tools_used:
        print(f"  (tools used: {', '.join(tools_used)})")
    print(f"Bot: {result['messages'][-1].content}\n")