# Conditional Edges
from typing import Literal

from langgraph.prebuilt import tools_condition

from agent.models import ContextSchema
from agent.typedicts import State
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage



def decide_start(state: State, runtime: Runtime[ContextSchema]) -> Literal["summarizer", "brain_node"]:
    message_threshold = runtime.context.message_threshold

    messages = state["messages"]

    result = None
    if len(messages) < message_threshold:
        result = "brain_node"
    if len(messages) >= message_threshold:
        result = "summarizer"
    return result

def decide_image_processing(state: State, runtime: Runtime[ContextSchema]) -> Literal["image_processor", "brain_node"]:
    messages = state["messages"]
    image_processing_required = False;
    
    for msg in messages:
        # Check if the tool output contains the magic signal
        if isinstance(msg, ToolMessage) and "VISUAL_INJECTION_64:" in msg.content:
            image_processing_required = True
            break
    result = None
    if image_processing_required:
        result = "image_processor"
    else:
        result = "brain_node"
    return result
def decide_after_brain(state: State) -> Literal["tools", "__end__"]:
    route = tools_condition(state)
    return route