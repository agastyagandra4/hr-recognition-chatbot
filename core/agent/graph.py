from pathlib import Path

from dotenv import load_dotenv

# Load GROQ_API_KEY before anything creates a Groq client
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from langchain_core.messages import AIMessage, SystemMessage  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from core.agent.tools import ALL_TOOLS, system_prompt  # noqa: E402
from core.guards.input_guard import GUARD_CONFIG, classify  # noqa: E402

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
SYSTEM_MESSAGE = SystemMessage(content=system_prompt())


class AgentState(MessagesState):
    """The conversation messages, plus the label the input guard gave the latest question."""
    guard_label: str


def input_guard_node(state: AgentState):
    """Label the newest question. If its rule says refuse, reply with the fixed message."""
    question = state["messages"][-1].content
    label, note = classify(question)
    rule = GUARD_CONFIG["labels"][label]

    if rule["action"] == "refuse":
        return {"guard_label": label, "messages": [AIMessage(content=rule["message"].strip())]}
    return {"guard_label": label}


def route_after_guard(state: AgentState):
    """Proceed to the agent, or stop here if the question was refused."""
    action = GUARD_CONFIG["labels"][state["guard_label"]]["action"]
    return "agent" if action == "proceed" else END


def agent_node(state: AgentState):
    """The AI reads the whole conversation and either answers or asks for a tool."""
    response = llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}


def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("input_guard", input_guard_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges("input_guard", route_after_guard, {"agent": "agent", END: END})
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())