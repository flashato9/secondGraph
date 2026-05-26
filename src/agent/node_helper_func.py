import copy
from datetime import datetime, timezone
from typing import List
import uuid

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.base import BaseStore

from agent.models import ConsolidationResult, ContextSchema, LLMConfiguration, MemoryExtraction, MemoryInsight, MemoryValue
from agent.tools import ALL_TOOLS
from agent.typedicts import State
from agent.types import LLM
from langgraph.runtime import Runtime

def get_message_flatten_text_content(message: AIMessage) -> AIMessage:
    """
    Standardizes AIMessage content for LangSmith readability.
    Joins multiple text blocks into one, while preserving tool calls, 
    images, or other non-text blocks.
    """
    if isinstance(message.content, str):
        message.content = [{"type": "text", "text": message.content}]
        return message

    if isinstance(message.content, list):
        new_content = []
        text_parts = []
        
        for block in message.content:
            # 1. Collect text blocks for merging
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            
            # 2. Keep other blocks (tool_use, image, etc.) as they are
            elif isinstance(block, dict):
                # If we have accumulated text, flush it before adding a non-text block
                if text_parts:
                    new_content.append({"type": "text", "text": "".join(text_parts)})
                    text_parts = []
                new_content.append(block)
            
            # 3. Handle raw strings mixed in lists
            elif isinstance(block, str):
                text_parts.append(block)

        # Final flush of accumulated text
        if text_parts:
            merged_text = "".join(text_parts)
            # If the ONLY thing in the message was text, LangChain prefers a string
            if not new_content:
                new_content = merged_text
            else:
                new_content.append({"type": "text", "text": merged_text})

        # Create the new message preserving all metadata/tool_calls
        new_message = copy.deepcopy(message)
        new_message.content = new_content
        return new_message

    return message

async def is_semantically_redundant(insight_content: str, namespace: tuple, store: BaseStore, threshold: float = 0.9) -> bool:
    """
    Checks if a similar insight already exists in the store to prevent 'bagel duplication'.
    """
    existing_matches = await store.asearch(
        namespace,
        query=insight_content,
        limit=1
    )

    if existing_matches:
        top_match = existing_matches[0]
        # If the vector similarity is higher than our threshold, it's a duplicate
        if top_match.score > threshold:
            return True
            
    return False

# Get LLM
async def get_llm(llm_config: LLMConfiguration, tools: list = ALL_TOOLS) -> LLM:
    model = ChatGoogleGenerativeAI(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=5
        )
    llm = model.bind_tools(tools)
    return llm

async def get_similar_in_category(
    insight: MemoryInsight, 
    namespace: tuple, 
    store: BaseStore, 
    threshold: float = 0.9, 
    limit: int = 10
):
    """
    Returns only memories that share the same category AND exceed the similarity threshold.
    """
    results = await store.asearch(namespace, query=insight.content, limit=20)
    
    # Combined filter: Category Match + Similarity Threshold
    return [
        res for res in results 
        if res.value.get("category") == insight.category.lower() 
        and res.score >= threshold
    ][:limit]

async def consolidate_and_verify(insight: MemoryInsight, lineage: list, config: dict) -> ConsolidationResult:
    """
    LLM determines the current stance and provides a confidence score to prevent summary drift.
    """
    llm = await get_llm(config, [])
    model = llm.with_structured_output(ConsolidationResult)
    
    lineage_text = "\n".join([f"- [{item.value.get('created_at')}] {item.value.get('content')}" for item in lineage])
    
    system_prompt = """
    You are a Memory Verification Expert. Analyze a new insight against historical records.
    Determine the definitive current stance. 
    Provide a 'confidence' score (0.0-1.0). If the history is contradictory or 
    the new insight is a total shift, lower the confidence.
    """
    
    prompt = f"NEW INSIGHT: {insight.content}\n\nHISTORY:\n{lineage_text}"
    return await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
        config={"tags": ["nostream"]}
        )

async def get_existing_categories(namespace: tuple, store: BaseStore) -> List[str]:
    """
    Retrieves the unique list of categories currently stored for this user.
    """
    # Pull a larger sample to ensure we capture the taxonomy
    # Note: If your store supports a specific 'distinct' query, use that instead.
    results = await store.asearch(namespace, query="", limit=50) 
    memories = [MemoryValue(**res.value) for res in results]
    categories = {
        res.category
        for res in memories 
        if res.category
    }
    return list(categories)

async def extract_new_insights(
    messages: list, 
    config: LLMConfiguration, 
    existing_categories: List[str]
) -> List[MemoryInsight]:
    """
    Distills insights while constraining categories to the existing taxonomy.
    """
    llm = await get_llm(config, [])
    model = llm.with_structured_output(MemoryExtraction)
    
    # Format categories for the prompt
    category_list = ", ".join(existing_categories) if existing_categories else "None yet"

    system_prompt = f"""
    You are a memory-distillation assistant for a Semantic OS. 
    Your goal is to extract NEW, meaningful insights from the provided conversation history.

    EXISTING TAXONOMY: {category_list}

    CONVERSATION STRUCTURE:
    - The messages below are provided in CHRONOLOGICAL ORDER (Earliest first, Latest last).
    - Each message includes a [YYYY-MM-DD HH:MM] timestamp.
    - Pay special attention to the LATEST messages, as they represent the user's most current state or updated preferences.

    GUIDELINES:
    1. REUSE CATEGORIES: If an insight fits into an existing category above, you MUST use it exactly.
    2. NEW CATEGORIES: Only create a new snake_case category if the insight absolutely does not fit.
    3. NO 'UNCATEGORIZED': Never use 'uncategorized'. Create a specific new label if needed.
    4. TYPE: 'fact' or 'user_preference'.
    5. CONTENT: Write a clear, standalone sentence. If a user's preference changed during this session, only extract the final, most recent preference.
    """
    
    extraction_result = await model.ainvoke(
        [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"CONVERSATION TO REVIEW:\n{messages}")
        ],
        config={"tags": ["nostream"]}
    )
    
    return extraction_result.insights

def get_last_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Starts at the end and works backward to find the most recent HumanMessage,
    then returns that message and all subsequent messages (the 'turn').
    """
    if not messages:
        return []

    # Find the index of the last HumanMessage
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            # Return from that HumanMessage to the very end
            return messages[i:]
            
    # Fallback: if no human message is found, return everything (or empty)
    return messages

async def memory_saver(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore):
    user_id = runtime.context.user_id
    active_ns = ("memories", user_id)
    stale_ns = ("stale_memories", user_id)
    thread_id = runtime.execution_info.thread_id
    llm_config = runtime.context.llm_configuration
    consine_similarity_threshold = runtime.context.consine_similarity_threshold

    last_messages = get_last_turn_messages(state["messages"])
    
    # 1. Fetch current taxonomy to prevent bloat
    existing_categories = await get_existing_categories(active_ns, store)
    
    new_insights = await extract_new_insights(last_messages, llm_config, existing_categories)

    for insight in new_insights:
        # 1. Targeted Search with Strict Category Filtering
        similar_items = await get_similar_in_category(insight, active_ns, store, threshold=consine_similarity_threshold)
        consolidation_threshold = 5
        is_redundant = len(similar_items) > consolidation_threshold

        if is_redundant:
            # 2. Consolidation with Verification
            result = await consolidate_and_verify(insight, similar_items, llm_config)
            
            if result.confidence >= 0.8:
                # 3. ARCHIVE: Move stale data & update references
                stale_ids = []
                for item in similar_items:
                    # Move to stale namespace
                    await store.aput(stale_ns, item.key, item.value)
                    # Delete from active
                    await store.adelete(active_ns, item.key)
                    stale_ids.append(item.key)
                    
                mem_obj = MemoryValue(
                    content=result.content,
                    type=insight.type,
                    category=insight.category.lower(),
                    thread_id=thread_id,
                    parent_references=stale_ids,
                    consolidation_reason=result.reasoning,
                    created_at=datetime.now(timezone.utc).isoformat()
                )
                # 4. INSERT SUMMARY: Point to archived lineage
                await store.aput(active_ns, str(uuid.uuid4()), mem_obj.to_dict())
                continue # Process next insight
        
        mem_obj = MemoryValue(
            content=insight.content,
            type=insight.type,
            category=insight.category.lower(),
            thread_id=thread_id,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        # 5. DEFAULT: Insert as new if not redundant OR if confidence was low
        await store.aput(active_ns, str(uuid.uuid4()), mem_obj.to_dict())

    return State(messages=[])

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
