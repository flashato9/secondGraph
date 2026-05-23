from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from agent.prompts import AGENT_PERSONA

class LLMConfiguration(BaseModel):
    "the configuration for the llm"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 1.0
    
class ToolsConfig(BaseModel):
    "the configuration for the tools"
    workspace_root: str = Field(
        default="./static/agent_files",
        description="The root directory for all file operations. Agents cannot access files outside this directory."
    )
    max_bytes: int = Field(
        default=50000,
        description="The maximum number of bytes to read from a file. If a file exceeds this, only the first max_bytes will be returned."
    )  
    current_relative_path: str = Field(
        default=".",
        description="The agent's current working directory relative to the workspace root. This allows the agent to 'navigate' within the sandbox."
    )
    execution_timeout : int = Field(
        default=30,
        description="The maximum time in seconds to allow a script to run when using the execute_file tool. This prevents infinite loops or long-running processes."
    )
    
class ContextSchema(BaseModel):
    llm_configuration: LLMConfiguration = LLMConfiguration()
    user_id: str = "default_user"
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
    message_threshold: int = Field(
        default=50,
        description="The number of recent messages to include in the context for decision making. Older messages will be summarized into the 'meaning' field of ToolResult."
    )
    tool_config: ToolsConfig = ToolsConfig()
    consine_similarity_threshold: float = Field(
        default=0.70,
        description="The default value to use for filtering good results for cosine_similariy"
    )

    
class ToolResult(BaseModel):
    """The universal response object for all agentic tools."""
    
    success: bool = Field(description="Whether the operation completed without errors")
    output: Any = Field(description="The primary data produced by the tool (e.g., file path, API JSON)")
    error: Optional[str] = Field(None, description="Detailed error message if success is False")
    meaning: str = Field(description="A human-readable summary for the agent's context")
    

    def to_state_update(self) -> dict:
        """Helper to format the result for LangGraph state updates."""
        return {
            "artifact_log": [self.model_dump()],
            "messages": [self.meaning]
        }
        
# Define a schema for the memory we want to extract
class MemoryExtraction(BaseModel):
    facts: list[str] = Field(description="New facts about the user or project")
    style_preferences: list[str] = Field(description="Preferences for how the AI should behave or code")
    
class MemoryInsight(BaseModel):
    content: str = Field(description="The actual fact or preference discovered.")
    type: Literal["fact", "user_preference"] = Field(description="The classification of the info.")
    category: str = Field(description="The grouping key (e.g., 'food_pref', 'coding_style'). Use snake_case.")

class MemoryExtraction(BaseModel):
    insights: List[MemoryInsight]   
    
class ConsolidationResult(BaseModel):
    content: str = Field(description="The new authoritative summary of the user's stance.")
    confidence: float = Field(description="Score from 0.0 to 1.0 on how accurately this represents the timeline.")
    reasoning: str = Field(description="Brief explanation of why this stance was chosen.")
    
class MemoryValue(BaseModel):
    content: str = Field(..., description="The distilled fact or preference statement.")
    type: str = Field(..., description="Classification: 'fact' or 'user_preference'.")
    category: str = Field(default="uncategorized", description="The snake_case grouping key for conflict resolution.")
    created_at: str = Field(
        default_factory=lambda: datetime.fromtimestamp(0, tz=timezone.utc).isoformat(),
        description="ISO 8601 timestamp of when the memory was captured or consolidated."
    )
    thread_id: str = Field(default="legacy_thread", description="The unique ID of the conversation thread.")
    parent_references: List[str] = Field(
        default_factory=list, 
        description="List of IDs in the stale_memories namespace that this entry summarizes."
    )
    consolidation_reason: str = Field(
        default="", 
        description="LLM's explanation for why this specific summary was created."
    )

    def to_dict(self):
        return self.model_dump()