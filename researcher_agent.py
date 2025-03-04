from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the graph builder
graph_builder = StateGraph(State)

# Initialize Tavily search tool
tool = TavilySearchResults(max_results=3)
tools = [tool]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def researcher(state: State):
    """Main researcher node that processes messages and decides whether to use tools."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add nodes to the graph
graph_builder.add_node("researcher", researcher)

# Add tool node using LangGraph's prebuilt ToolNode
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Add edges with conditional routing
graph_builder.add_conditional_edges(
    "researcher",
    tools_condition,
    {
        "continue": "researcher",  # Continue conversation
        "tool": "tools",  # Use tool
    },
)

# Add the final edge from tools back to researcher
graph_builder.add_edge("tools", "researcher")

# Compile the graph
graph = graph_builder.compile() 