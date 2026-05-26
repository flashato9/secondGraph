# Graph Nodes
import asyncio
import uuid

from langgraph.graph import END

from agent.conditional_edges import decide_after_brain
from agent.models import ContextSchema
from agent.node_and_tool_helper_func import get_relevant_memories
from agent.node_helper_func import get_llm, get_message_flatten_text_content, get_sanitized_messages, memory_saver
from agent.reducers import robust_message_reducer
from agent.typedicts import State
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.store.base import BaseStore


async def summarizer(state: State, runtime: Runtime[ContextSchema]) -> State:
    llm_config = runtime.context.llm_configuration
    llm_with_tools = await get_llm(llm_config, tools=[]) # No tools for summarization step
    message_threshold = runtime.context.message_threshold    
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
    ai_response = await llm_with_tools.ainvoke(
                                                llm_input,
                                                config={"tags": ["nostream"]}
                                                )
    ai_response = get_message_flatten_text_content(ai_response)
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

async def brain(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore) -> State:
    # 1. Setup and context gathering
    llm_config = runtime.context.llm_configuration
    user_messages = get_sanitized_messages(state["messages"])
    namespace = ("memories", runtime.context.user_id)
    
    memory_context = ""

    # 2. Semantic Memory Retrieval (Only on new Human turns)
    if not isinstance(user_messages[-1], ToolMessage):
        query_text = next((str(m.content) for m in reversed(user_messages) if isinstance(m, HumanMessage)), "")
        if query_text:
            memory_context = await get_relevant_memories(query_text, namespace, store)
            
    # 3. Message Construction
    system_content = f"{runtime.context.persona}\n{memory_context}"
    final_messages = [SystemMessage(content=system_content)] + user_messages

    # 4. LLM Execution
    llm_with_tools = await get_llm(llm_config)
    ai_message = await llm_with_tools.ainvoke(
                                                final_messages,
                                                # config={"tags": ["nostream"]}
                                             )
    
    # Flatten multi-block content for LangSmith/State consistency
    ai_message = get_message_flatten_text_content(ai_message)
    result_state = State(messages=[ai_message])
    # run memory saver asynchronously so we don't block the agent's response while we do consolidation and store updates
    intermediate_state = State(messages=robust_message_reducer(state["messages"], [ai_message]))
    if decide_after_brain(intermediate_state) == END:
        asyncio.create_task(memory_saver(intermediate_state, runtime, store=store))
    return result_state

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