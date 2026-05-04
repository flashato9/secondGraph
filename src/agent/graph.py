"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import inspect
from math import remainder
from os import system
from re import S
from typing import Annotated, Literal
from urllib import response
import uuid

import aiofiles
from dotenv import load_dotenv
import lancedb
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_core.messages import convert_to_messages
from mypy import state
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, data
from langgraph.runtime import Runtime
from pathlib import Path
from langgraph.graph.message import add_messages
from agent.prompts import AGENT_PERSONA
from langgraph.checkpoint.memory import MemorySaver
from agent.tools import ALL_TOOLS, run_python_task, search_memory
from agent.types import LLM
from langchain_core.messages import convert_to_messages
import uuid
from langchain_core.load import load

#Converters
def convert_to_valid_messages(meassges: list[AnyMessage] | list[dict]) -> list[AnyMessage]:
    """
    Converts serialized constructor dicts or standard dicts 
    into valid LangChain Message objects with guaranteed IDs.
    """
    result = []
    # 1. Handle the LangChain 'constructor' serialization format
    for data in meassges:
        message = None
        if isinstance(data, dict) and data.get("type") == "constructor":
            try:
                message = load(data)
            except Exception:
                kwargs = data.get("kwargs", {})
                message = convert_to_messages([kwargs])[0]
        else:
            message = convert_to_messages([data])[0]
        result.append(message)
    for msg in result:
        # 2. Ensure every message has a unique ID
        if not getattr(msg, 'id', None):
            try: 
                msg.id = str(uuid.uuid4())
            except Exception as e:
                print(f"Error occurred while generating ID for message: {e}")
    return result
#Reducers
def robust_message_reducer(left: list[AnyMessage], right: list[AnyMessage] | list[dict]) -> list[AnyMessage]:
    """Handles standard message updates AND serialized constructor dicts from Fork/Redo."""
    processed_right = []
    processed_left = convert_to_valid_messages(left)
    processed_right = convert_to_valid_messages(right)
    return add_messages(processed_left, processed_right)

# Classes - State, LLMConfiguration, ContextSchema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], robust_message_reducer]

       
class LLMConfiguration(BaseModel):
    "the configuration for the llm"
    model_name: str = "gemini-flash-lite-latest"
    temperature: float = 1.0

class ContextSchema(BaseModel):
    llm_configuration: LLMConfiguration = LLMConfiguration()
    persona: str = Field(
        default = (
           AGENT_PERSONA
           ),
        description = "The persona for the assistant",
        json_schema_extra={
            "langgraph_nodes": ["brain_node"],
            "langgraph_type": "prompt"
        }
    )

# Get LLM
async def get_llm(llm_config: LLMConfiguration, tools: list = ALL_TOOLS) -> LLM:
    model = ChatGoogleGenerativeAI(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    llm = model.bind_tools(tools)
    return llm

#Node Helper Functions
def get_sanitized_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Sanitizes history for Gemini without modifying the global state.
    - Merges consecutive HumanMessages.
    - Merges consecutive SystemMessages.
    - Removes 'Orphaned' AIMessages (tool calls with no response).
    - Ensures ToolMessages follow the specific AI turn that called them.
    """
    if not messages:
        return []

    processed = []

    pre_processed = []

    # 1. Identify the first non-system message
    first_msg_index = 0
    while first_msg_index < len(messages) and isinstance(messages[first_msg_index], SystemMessage):
        pre_processed.append(messages[first_msg_index])
        first_msg_index += 1
        
    # 2. Check if the NEXT message is an illegal Tool Call
    if first_msg_index < len(messages):
        target = messages[first_msg_index]
        if isinstance(target, AIMessage) and (target.tool_calls or target.additional_kwargs.get("function_call")):
            # INJECT a dummy Human Message to satisfy Gemini's protocol
            pre_processed.append(HumanMessage(content="Continuing previous task..."))
            print("Self-Healing: Injected 'Ghost' Human Message to fix summary truncation.")

    # 3. Add the rest of the messages
    pre_processed.extend(messages[first_msg_index:])
    
    for i, msg in enumerate(pre_processed):
        # 1. Handle consecutive SystemMessages (Merge)
        if isinstance(msg, SystemMessage) and processed and isinstance(processed[-1], SystemMessage):
            processed[-1] = SystemMessage(content=f"{processed[-1].content}\n\n{msg.content}")
            continue

        # 2. Handle consecutive HumanMessages (Merge)
        if isinstance(msg, HumanMessage) and processed and isinstance(processed[-1], HumanMessage):
            processed[-1] = HumanMessage(content=f"{processed[-1].content}\n\n{msg.content}")
            continue

        # 3. Handle AIMessages with Tool Calls
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Look ahead: is the next message a ToolMessage?
            # We check the original 'messages' list for this check
            has_tool_resp = (i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage))
            
            if not has_tool_resp:
                # This is an orphaned tool call. Gemini will 400.
                # We skip this message entirely to 'heal' the sequence.
                print(f"Self-Healing: Skipping orphaned tool call from AI (ID: {getattr(msg, 'id', 'unknown')})")
                continue

        # 4. Handle ToolMessages (Ensure they don't follow a HumanMessage)
        if isinstance(msg, ToolMessage) and processed and isinstance(processed[-1], HumanMessage):
            # This is a rare edge case if your image_processor injected a HumanMessage 
            # between a tool call and its response. We move the ToolMessage up.
            human_msg = processed.pop()
            processed.append(msg)
            processed.append(human_msg)
            continue

        processed.append(msg)

    return processed

# Graph Nodes
async def summarizer(state: State, runtime: Runtime[ContextSchema]) -> State:
    llm_config = runtime.context.llm_configuration
    llm_with_tools = await get_llm(llm_config, tools=[]) # No tools for summarization step
    message_threshold = 15
    number_messages_to_keep = int(message_threshold*0.45)
    messages = state["messages"]
    cutoff_index = len(messages) - number_messages_to_keep

    # SAFETY: Move the cutoff back if we are in the middle of a Tool Call
    while cutoff_index > 0:
        # If the message at the cutoff is an AI Tool Call or a Tool Message, 
        # move the cutoff back so we don't break the chain.
        if isinstance(messages[cutoff_index], (ToolMessage, AIMessage)):
            cutoff_index -= 1
        else:
            break

    result = None
    if len(messages) < message_threshold:
        result = State(messages=[])
    if len(messages) >= message_threshold:
        
        system_prompt = SystemMessage(content="You are a helpful assistant that summarizes conversations, preserving all file paths mentioned.")
        summary_prompt = HumanMessage(content="""
                    Summarize the previous conversation and return a concise summary that captures all important details, especially any file paths or tool outputs. 
                    Be sure to retain any information that might be relevant for future context. 
                    The summary should be brief but comprehensive.
                    The summary should be 10 sentences long maximum.
                    The summary should have the following format:
                    The following content is a summary of the conversation prior: <insert summary here>
                                      """)
        past_messages = messages[:cutoff_index] 
        llm_input = past_messages + [system_prompt] + [summary_prompt]
        ai_response = await llm_with_tools.ainvoke(llm_input)
        ai_response_as_syastem_message = SystemMessage(content=ai_response.content[0]["text"])
        ai_response_as_syastem_message.id = str(uuid.uuid4())
        removed_past_messages = [RemoveMessage(id=msg.id) for msg in messages[:cutoff_index]]
        removed_messages_to_keep = [RemoveMessage(id=msg.id) for msg in messages[cutoff_index:]]
        messages_to_keep_with_new_id = []
        for msg in messages[cutoff_index:]:
            new_msg = msg.model_copy()
            new_msg.id = str(uuid.uuid4())
            messages_to_keep_with_new_id.append(new_msg)
        messages = [ai_response_as_syastem_message] + removed_past_messages + removed_messages_to_keep + messages_to_keep_with_new_id
        result = State(messages=messages)
    return result

async def brain(state: State, runtime: Runtime[ContextSchema]) -> State:
    llm_config = runtime.context.llm_configuration
    model_input = get_sanitized_messages(state["messages"])
    
    # LLM Persona should be light and loaded per context. LLM can overwrite persona. but persona should be light.
    # Additional Funcitonality can be added
    # Ensure the file exists on startup
    
    final_messages = model_input;
    if not isinstance(model_input[-1], ToolMessage):
        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(None, PROMPT_PATH.exists)
        if not exists:
            await loop.run_in_executor(
                None, 
                lambda: PROMPT_PATH.write_text("You are a helpful AI assistant.")
            )
        current_persona = None
        async with aiofiles.open(PROMPT_PATH, mode='r') as f:
            current_persona = await f.read() # You must await the .read() call specifically
        llm_persona = SystemMessage(content=current_persona)
        final_messages = [llm_persona] + final_messages
    llm_with_tools = await get_llm(llm_config)
    ai_message = await llm_with_tools.ainvoke(final_messages)
    
    # After the response, we index the turn into LanceDB for Tier 3 memory
    # text_to_index = f"User: {model_input[-1].content}\nAI: {ai_message.content}"
    # tbl = get_or_create_table()
    # vector = await embeddings_model.aembed_query(text_to_index)
    # loop = asyncio.get_event_loop()
    # await loop.run_in_executor(None, lambda: tbl.add([{"text": text_to_index, "vector": vector}]))
    
    result = State(
        messages = [ai_message]
    )
    return result

async def image_processor(state: State, runtime: Runtime[ContextSchema]) -> State:
    """
    Scans message history for the VISUAL_INJECTION_64 signal 
    and converts text-based Base64 tool outputs into vision-capable blocks.
    """
    messages = state["messages"]
    refined_messages = []
    
    for msg in messages:
        # Check if the tool output contains the magic signal
        if isinstance(msg, ToolMessage) and "VISUAL_INJECTION_64:" in msg.content and "VISUAL_INJECTION_64: <your_base64_string>" not in msg.content:
            try:
                # Extract the Base64 payload
                parts = msg.content.split("VISUAL_INJECTION_64:")
                b64_data = parts[1].strip()
                
                # Create the formatted vision message
                vision_msg = HumanMessage(
                    content=[
                        {"type": "text", "text": "REPL Image Processing Complete. Visual context attached below:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                    ]
                )
                refined_messages.append(vision_msg)
            except Exception as e:
                # Fallback if the string was malformed
                refined_messages.append(HumanMessage(content=f"Error decoding visual injection: {str(e)}"))
                refined_messages.append(msg)

    result = State(messages=refined_messages)   
    return result


# Conditional Edges
def decide_start(state: State, runtime: Runtime[ContextSchema]) -> Literal["summarizer", "brain_node"]:
    message_threshold = 15

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

# Graph Definition
def get_graph():
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
        tools_condition, 
        {
            "tools": "tools", # If tool called, go to 'tools' node
            END: END          # If no tool called, finish
        }
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
# # Persistence Layer
# def get_or_create_table():
#     table_name = "chat_history"
#     # Check if the table exists in the current database
#     if table_name in db.table_names():
#         return db.open_table(table_name)
#     else:
#         # Create the table with an initial dummy record to define the schema
#         # The schema is inferred from this first entry
#         return db.create_table(
#             table_name, 
#             data=[{"text": "initial_seed", "vector": [0.0] * 3072}] # 3072 is standard for Google text-embedding-2
#         )


load_dotenv()
PROMPT_PATH = Path("C:\\Users\\Ato_K\\Documents\\programming\\RoomLogic\\.agent_data\\system_prompt.md")

graphName = "Agent"
graph = ( get_graph()
            .compile(
                name=graphName
        )
)

# tools CRUD + Execute
#   create files on the file system.
#   read files on the file system.
#   update files on the file system.
#   delete files on the file system.
#   execute bash scripts on the system
#   create new tools and add it to the agent. (how do I do this?
# We are reaching into self modifying code territory.



