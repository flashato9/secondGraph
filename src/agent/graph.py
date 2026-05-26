"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
import debugpy
from dotenv import load_dotenv
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from agent.conditional_edges import decide_after_brain, decide_image_processing, decide_start
from agent.nodes import brain, image_processor, summarizer
from agent.tools import ALL_TOOLS
from agent.typedicts import State

from agent.models import ContextSchema
# Graph Definition
def get_graph():
    try:
        # 0.0.0.0 is required inside Docker!
        debugpy.listen(("0.0.0.0", 5679))
    except Exception:
        pass # Prevents the worker crash we saw earlier

    workflow = StateGraph(State, context_schema=ContextSchema)
    
    # Nodes
    workflow.add_node("brain_node", brain)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("image_processor", image_processor)
    workflow.add_node("tools", ToolNode(ALL_TOOLS)) # 'tools' is your list of @tool functions
    
    # Edges
    workflow.add_conditional_edges(
        START,
        decide_start,
        {
            "summarizer": "summarizer",
            "brain_node": "brain_node"
        }
    )
    workflow.add_edge("summarizer", "brain_node")
    workflow.add_conditional_edges(
        "brain_node",
        # This helper function automatically checks if the LLM called a tool
        decide_after_brain, 
    )
    workflow.add_conditional_edges(
        "tools",
        decide_image_processing,
        {
            "image_processor": "image_processor",
            "brain_node": "brain_node"
        } 
    )
    workflow.add_edge("image_processor", "brain_node")  # After image processing, go back to brain
    return workflow

# Embedding Function
# 2. Add this wrapper function specifically for LangGraph API
async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """LangGraph worker will call this async function to embed text."""
    return await embedding_object.aembed_documents(texts)

# Compiled Graph Definition
def get_compiled_graph() -> StateGraph:
    graphName = "Agent"
    result = (
        get_graph()
            .compile(
                name=graphName,
            )
    )
    return result;

#LOAD Variables - embedding object, graph
load_dotenv()
embedding_object = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=768 
)
graph = get_compiled_graph()
# tools CRUD + Execute
#   create files on the file system.
#   read files on the file system.
#   update files on the file system.
#   delete files on the file system.
#   execute bash scripts on the system
#   create new tools and add it to the agent. (how do I do this?
# We are reaching into self modifying code territory.



