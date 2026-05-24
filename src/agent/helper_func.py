import copy
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.base import BaseStore

from agent.models import ConsolidationResult, LLMConfiguration, MemoryExtraction, MemoryInsight, MemoryValue
from agent.node_helper_func import get_llm
from agent.tools import ALL_TOOLS
from agent.types import LLM

def get_message_flatten_text_content(message: AIMessage) -> AIMessage:
    """
    Standardizes AIMessage content for LangSmith readability.
    Joins multiple text blocks into one, while preserving tool calls, 
    images, or other non-text blocks.
    """
    if isinstance(message.content, str):
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