from typing import Annotated, Dict, Union

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def get_config():
    """Configure and return the graph."""
    # Initialize the graph builder
    graph_builder = StateGraph(State)

    # Initialize Tavily search tool
    tool = TavilySearchResults(max_results=3)
    tools = [tool]

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def researcher(state: State) -> Dict:
        """Main researcher node that processes messages and decides whether to use tools."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        if isinstance(response, AIMessage):
            return {"messages": [response]}
        return {"messages": [AIMessage(content=str(response))]}

    def should_continue(state: State) -> Union[str, END]:
        """Determine if we should end or continue using tools."""
        messages = state["messages"]
        if not messages:
            return "tools"
            
        last_message = messages[-1]
        # If the last message was from the AI and doesn't require tools
        if isinstance(last_message, AIMessage) and tools_condition({"messages": messages}) != "tools":
            return END
        return "tools"

    # Add nodes to the graph
    graph_builder.add_node("researcher", researcher)

    # Add tool node using LangGraph's prebuilt ToolNode
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    # Set the entry point first
    graph_builder.set_entry_point("researcher")

    # Add edges with conditional routing
    graph_builder.add_conditional_edges(
        "researcher",
        should_continue,
        {
            END: END,  # End the conversation
            "tools": "tools",  # Use tool
        },
    )

    # Add the final edge from tools back to researcher
    graph_builder.add_edge("tools", "researcher")

    # Compile the graph
    return graph_builder.compile()


# Create the graph instance
graph = get_config()

# Define the default config
config = {
    "messages": [HumanMessage(content="Hello, I need help with research.")]
}

# Export for LangGraph API
__all__ = ["graph"]