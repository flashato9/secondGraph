import uuid

from langchain_core.messages import convert_to_messages
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages
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