"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import remainder
from os import system
from re import S
from typing import Annotated, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.types import Command, interrupt
from langchain_core.messages import convert_to_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import base64
import aiofiles
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.runtime import Runtime
from pathlib import Path

from agent.prompts import AGENT_PERSONA
from langgraph.checkpoint.memory import MemorySaver

from agent.tools import delete_image, get_image_database, getPicture, upload_latest_image

load_dotenv()
tools = [getPicture,upload_latest_image, get_image_database,delete_image]

def add_reducer(existing_message:list[AnyMessage], new_message: list[AnyMessage]) -> list[AnyMessage]:
    # note to self, the reducer applies to a member of the state. It takes two of the same type of memeber and generates one.
    add_messages_result = add_messages(existing_message,new_message)
    result = [elem for elem in add_messages_result]
    return result

async def get_llm(llm_config: LLMConfiguration):
    model = ChatGoogleGenerativeAI(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    llm_with_tools = model.bind_tools(tools)
    return llm_with_tools


def consolidate_disjoint_text_content(message:AIMessage):
    #note to self: the content here is also a list of content blocks. A multi-modal model accepts as input a list of content blocks and returns as output a list of content blocks.
    content = message.content
    result_content = None
    all_text_content = []
    non_text_content = []
    if not isinstance(content,list):
        result_content = content
    else:
        for elem in content:
            if isinstance(elem,str):
                all_text_content.append(elem)
            elif isinstance(elem,dict) and "text" in elem:
                all_text_content.append(elem["text"])
            else:
                non_text_content.append(elem)
        all_text_content = [{"type":"text","text":"".join(all_text_content)}]
        result_content = all_text_content + non_text_content
    message.content = result_content
    return message
    
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_reducer]
    
class LLMConfiguration(BaseModel):
    "the configuration for the llm"
    model_name: str = "gemini-flash-lite-latest"
    temperature: float = 1.0

class ContextSchema(BaseModel):
    #note to self: only pydantic basemodel allows default values. TypedDict does not generate default values.
    #note to self: you can use the json_schema extra and use doc string and description to better explain the configuration and it can also be recursive.
    #note to self: its typically best to use simple types where best and complicate things if they work. But stnadard types like string, int, float. Those work well
    #note to self: you can create node scoped assistant configurations.
    llm_configuration: LLMConfiguration = LLMConfiguration()
    persona: str = Field(
        default = (
           AGENT_PERSONA
           ),
        description = "The persona for the assistant",
        json_schema_extra={
            "langgraph_nodes": ["room_evaluator"],
            "langgraph_type": "prompt"
        }
    )

async def room_evaluator(state: State, runtime: Runtime[ContextSchema]) -> State:
    #note to self: a node takes in a state and outputs a state.
    #note to self: a Pydantic BaseModel requires you to access to object via dot operator and not using square brackets as you would do with TypedDict  
    model_input = state["messages"]
    llm_persona = SystemMessage(content= runtime.context.persona)
    llm_config = runtime.context.llm_configuration
    llm_with_tools = await get_llm(llm_config)
    ai_message = await llm_with_tools.ainvoke([llm_persona] + model_input)
    ai_message = consolidate_disjoint_text_content(ai_message)
    result = State(
        messages = [ai_message]
    )
    return result

async def toolApprover(state:State, runtime:Runtime[ContextSchema]) -> Literal["tools","prompt_editor", "prompt_rejecter","__end__"]:
    is_last_message_contains_tool_call = len(state["messages"][-1].tool_calls)>0
    if is_last_message_contains_tool_call:
        interrupt_return_message = interrupt({
            "query": "Do you approve the tool call?",
            "answer_options": "Approve/Edit/Reject",
            "tool_call":state["messages"][-1].tool_calls
        })
        if interrupt_return_message == "Approve":
            return "tools"
        if interrupt_return_message == "Edit":
            return "prompt_editor"
        if interrupt_return_message == "Reject":
            return "prompt_rejecter"
    else:
        return END

async def prompt_editor(state: State, runtime: Runtime[ContextSchema]) -> State:
    messages = state["messages"]

    previous_message = messages[-2].content
    new_message = interrupt({
    "query": f"""
        The previous prompt was - {previous_message}.
        Enter the prompt to replace this prompt.
    """
    })
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages[-2:]]
    message_to_add = HumanMessage(
        content = new_message
    )
    return State(
        messages = messages_to_remove + [message_to_add]
    )

async def prompt_rejecter(state: State, runtime: Runtime[ContextSchema]) -> State:
    messages = state["messages"]
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages[-2:]]
    return State(
        messages = messages_to_remove
    )
def get_graph():
    res = (StateGraph(State, context_schema =ContextSchema)
    .add_node("room_evaluator",room_evaluator)
    .add_node("prompt_editor",prompt_editor)
    .add_node("prompt_rejecter",prompt_rejecter)
    .add_node("tools",ToolNode(tools))
    .add_edge(START, "room_evaluator")
    .add_edge("tools","room_evaluator")
    .add_conditional_edges("room_evaluator",toolApprover)
    .add_edge("prompt_editor","room_evaluator")
    .add_edge("prompt_rejecter",END)
    )
    return res

graphName = "Room Logic Agent"
graph = ( get_graph()
            .compile(
                name=graphName
        )
)
