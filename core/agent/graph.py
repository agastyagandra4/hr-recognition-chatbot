from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from core.agent.tools import ALL_TOOLS, system_prompt

load_dotenv()  # reads GROQ_API_KEY from your .env file

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
SYSTEM_MESSAGE = SystemMessage(content=system_prompt())


def agent_node(state: MessagesState):
    """The AI reads the whole conversation and either answers or asks for a tool."""
    response = llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}


def build_agent():
    builder = StateGraph(MessagesState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # tool requested? -> tools, else -> end
    builder.add_edge("tools", "agent")                       # after tools, back to the agent

    return builder.compile(checkpointer=MemorySaver())