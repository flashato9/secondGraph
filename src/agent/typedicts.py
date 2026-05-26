
# Classes - State, LLMConfiguration, ContextSchema
from typing_extensions import Annotated

from langchain_protocol import TypedDict

from agent.reducers import robust_message_reducer
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], robust_message_reducer]