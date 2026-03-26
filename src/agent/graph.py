"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
import random 

class State(TypedDict):
    message: str


async def roomEvaluator(state: State) -> State:
    result = State(
        message = "this is a result"
    )
    return result

async def decideRoom(state: State) -> Literal["room1Observer","room2Observer"]:
    result = None
    if "room1" in state["message"]:
        result = "room1Observer"
    else:
        result = "room2Observer"
    return result

async def room1Observer(state: State) -> State:
    result = State(
        message = "You entered and left room 1."
    )
    return result

async def room2Observer(state: State) -> State:
    result = State(
        message = "You entered and left room 2."
    )
    return result

# Define the graph
graphName = "Room Evaluator Agent"
graph = (
    StateGraph(State)
    .add_node(roomEvaluator)
    .add_node(room1Observer)
    .add_node(room2Observer)
    .add_edge(START, "roomEvaluator")
    .add_conditional_edges("roomEvaluator",decideRoom)
    .add_edge("room1Observer", "roomEvaluator")
    .add_edge("room2Observer", "roomEvaluator")
    .add_edge("roomEvaluator",END)
    .compile(name=graphName)
)
