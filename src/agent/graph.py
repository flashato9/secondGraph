"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from math import remainder
from re import S
from typing import Annotated, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from typing_extensions import TypedDict
import base64
import aiofiles
import json
from langchain_google_genai import ChatGoogleGenerativeAI


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_reducer]

def add_reducer(existing_message:list[AnyMessage], new_message: list[AnyMessage]) -> list[AnyMessage]:
    # note to self, the reducer applies to a member of the state. It takes two of the same type of memeber and generates one.
    result = (existing_message if existing_message else []) + (new_message if new_message else [])
    return result

async def getPicture(path: str) -> list:
    #note to self: the input to a tool is the input that your function should take. The LLM will figureout the input to use for the function. Just make sure to include type hints, doc string
    #note to self: tool functions return a list of content blocks. The ToolNode() will assign this list the content member of a ToolMessage
    #note to self: each element in the content block is fed to the model in the order of its specification. Each content block has a type and content.
    #note to self: when you return the image_url content the model will embedd the image and load it into its context for memory.
    """Get the bytes to an image.

    Args:
        path: path to image file
    Returns:
        A human message containing the image bytes.
    """
    image_bytes = None
    async with aiofiles.open(path, mode='rb') as f:
        image_bytes = await f.read()
    image64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = "image/png"
    content =[
        # {
        #     "type": "text",
        #     "text": "Describe the contents of this image."
        # },
        # {
        #     "type":"image",
        #     "base64": image64,
        #     "mime_type": mime_type,
        # },
        {
            "type": "image_url",
            "image_url": {
                    "url": f"data:image/jpeg;base64,{image64}"
                },
        },
        
    ]
    return content

load_dotenv()
tools = [getPicture]
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_with_tools = model.bind_tools(tools)

def compress_text_content(message:AIMessage):
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

async def roomEvaluator(state: State) -> State:
    #note to self: a node takes in a state and outputs a state.
    model_input = state["messages"]
    ai_message = await llm_with_tools.ainvoke(model_input)
    ai_message = compress_text_content(ai_message)
    result = State(
        messages = [ai_message]
    )
    return result
graphName = "Room Evaluator Agent"
graph = ( 
    StateGraph(State)
    .add_node("roomEvaluator",roomEvaluator)
    .add_node("tools",ToolNode([getPicture]))
    .add_edge(START, "roomEvaluator")
    .add_edge("tools","roomEvaluator")
    .add_conditional_edges("roomEvaluator",tools_condition)
    .compile(name=graphName)
)
