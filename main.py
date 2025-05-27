import os
from dotenv import load_dotenv  # ✅ Add this line

from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tool import get_all_tools

# ✅ Load .env variables
load_dotenv()

# ✅ Safely assign only if value exists
tavily_api_key = os.getenv("TAVILY_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Tool and LLM setup
tools = get_all_tools()
llm = ChatGroq(model='qwen-qwq-32b').bind_tools(tools=tools)

# LangGraph State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Build LangGraph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()
