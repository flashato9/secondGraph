import json
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path  # Use the standard library
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Any, List, Optional
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime
from langchain_tavily import TavilySearch
from langgraph.store.base import BaseStore


from agent.node_and_tool_helper_func import get_relevant_memories
from agent.models import ContextSchema, LLMConfiguration, ToolsConfig, ToolResult

# Path to your ignored data folder
DATA_DIR = r"C:\Users\Ato_K\Documents\programming\SemanticOS\.agent_data"
JSONL_LOG_PATH = os.path.join(DATA_DIR, "terminal_history.jsonl")

def log_to_jsonl(input_code, output, reasoning, status="success"):
    """Appends a structured entry to the JSONL ledger."""
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "intent": reasoning,
        "input": input_code,
        "output": output.strip() if output else "No output"
    }
    
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

@tool
def run_python_task(code: str, reasoning: str) -> str:
    """
    Executes Python code in a fresh, stateless environment.
    Use this for file management, running scripts, and system tasks.
    
    Args:
        code: The Python code to execute.
        reasoning: Why are you running this? (Saved to the ledger).
    """
    # Instantiate a fresh interpreter for every call (Stateless)
    repl = PythonREPL()
    result = None
    try:
        # Execute the code and capture stdout
        result = repl.run(code)
    except Exception as e:
        # Catch system-level or execution-level crashes
        error_msg = f"Runtime Error: {str(e)}"
        log_to_jsonl(code, error_msg, reasoning, status="crash")
        return error_msg
    
    # Check if the REPL returned an error string (some REPLs return errors in the output)
    if "Traceback" in result or "Error" in result:
        log_to_jsonl(code, result, reasoning, status="error")
    else:
        log_to_jsonl(code, result, reasoning, status="success")
        
    return result

    
# db = lancedb.connect("./.agent_data/vectors")
# table_name = "chat_history"
# embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
@tool
async def search_memory(
    query: str, 
    runtime_config: ToolRuntime[ContextSchema]
) -> str:
    """
    Search your long-term memory for specific facts, preferences, or 
    past interactions that aren't in the current conversation.
    
    Args:
        query: The search term or semantic description of the memory to find.
    Returns:
        A structured response containing:   
        - success: Whether the memory read operation was successful.
        - output: If successful, includes the memory content.
        - error: If failed, a detailed error message.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.
    
    """
    store = runtime_config.store
    user_id = runtime_config.context.user_id
    consine_similarity_threshold = runtime_config.context.consine_similarity_threshold
    
    namespace = ("memories", user_id)

    try:
        # 1. Reuse the core helper function
        # We increase the limit slightly for tool calls (10 vs 5) to give the agent more to work with
        memory_text = await get_relevant_memories(
            query=query,
            namespace=namespace,
            store=store,
            threshold=consine_similarity_threshold,
            limit=10
        )

        # 2. Handle Empty Results
        if not memory_text:
            result = ToolResult(
                success=True,
                output=None,
                meaning=f"No relevant memories found for the query: '{query}'."
            )
        else:
            # 3. Success Feedback
            result = ToolResult(
                success=True,
                output={"content": memory_text},
                meaning=f"Relevant historical context retrieved for: '{query}'."
            )

    except Exception as e:
        # 4. Error Handling
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"An error occurred while searching memory: {str(e)}"
        )

    return result.to_state_update()

@tool
def create_file(file_path: str, content: str, runtime_config: ToolRuntime[ContextSchema]) -> str:
    """
    Standardizes how an agent persists data to the workspace.
    
    Args:
        file_path: Relative path where the file should be created within the agent's allowed directory.
        content: The string content to write into the file.
    Returns:
        A structured response containing:   
        - success: Whether the write operation was successful.
        - output: If successful, includes file path and size.
        - error: If failed, a detailed error message.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.
    """
    workspace_root = runtime_config.context.tool_config.workspace_root
    try:
        # 1. Path Validation & Security (Preventing Path Traversal)
        # Ensure the agent isn't trying to write to /etc/passwd or system roots.
        root = Path(workspace_root).resolve()
        target_path = (root / file_path).resolve()

        if not str(target_path).startswith(str(root)):
            raise PermissionError(f"Path traversal detected: {file_path} is outside workspace.")

        # 2. Directory Persistence
        # Check if the parent folders exist; if not, create them (mkdir -p logic).
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. Atomic Write
        # Write content and ensure buffers are flushed.
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno()) # Ensure it's physically on disk
        
        # 4. Feedback Loop
        # Return a confirmation: "File successfully created at [path]. Size: [X] bytes."
        result = ToolResult(
            success=True,
            output={"path": str(target_path), "size_bytes": len(content)},
            meaning=f"File '{file_path}' successfully created in the workspace."
        )
    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Failed to create file '{file_path}': {str(e)}"
        )
    return result.to_state_update()
@tool
def read_file(filename: str, runtime_config: ToolRuntime[ContextSchema]) -> dict:
    """
    Standardizes how an agent reads files from the workspace, with built-in guardrails.

    Args:
        filename: Relative path to the file within the agent's allowed directory.
        runtime_config: The configuration for the tool execution, which may include:
            - workspace_root: The root directory for the agent's file operations.
            - max_bytes: The maximum number of bytes to read (to prevent memory overload).
    Returns:
        A structured response containing:   
        - success: Whether the read operation was successful.
        - output: If successful, includes file content (or a portion of it), file path, and total size.
        - error: If failed, a detailed error message.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.
    """
    
    try:
        # 1. Path Validation & Security
        workspace_root = runtime_config.context.tool_config.workspace_root
        max_bytes = runtime_config.context.tool_config.max_bytes
        root = Path(workspace_root).resolve()
        target_path = (root / filename).resolve()

        if not str(target_path).startswith(str(root)):
            raise PermissionError(f"Access Denied: {filename} is outside the sandbox.")

        if not target_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found.")

        # 2. Size Validation (The "Guardrail")
        file_size = target_path.stat().st_size
        if file_size > max_bytes:
            # We return a partial read or an error to force the agent to use a 'chunked' reader
            content = target_path.read_text(encoding="utf-8")[:max_bytes]
            meaning = (f"File '{filename}' is too large ({file_size} bytes). "
                       f"I have read the first {max_bytes} bytes for you.")
        else:
            content = target_path.read_text(encoding="utf-8")
            meaning = f"Successfully read file: {filename}"

        # 5. Feedback Loop & State Sync
        result = ToolResult(
            success=True,
            output={"content": content, "path": str(target_path), "total_size": file_size},
            meaning=meaning
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Failed to read file '{filename}': {str(e)}"
        )

    return result.to_state_update()

    
@tool
def update_file(filename: str, content: str, append: bool, runtime_config: ToolRuntime[ContextSchema]) -> dict:
    """
    Standardizes how an agent updates (overwrites or appends to) files in the workspace, with built-in guardrails.
    Args:
        filename: Relative path to the file within the agent's allowed directory.
        content: The string content to write into the file.
        append: If True, content will be appended; if False, the file will be overwritten.
        runtime_config: The configuration for the tool execution, which may include:
            - workspace_root: The root directory for the agent's file operations.
    Returns:    
        A structured response containing:
        - success: Whether the update operation was successful.
        - output: If successful, includes file path, operation type, and new file size. 
        - error: If failed, a detailed error message.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.

    """
    workspace_root = runtime_config.context.tool_config.workspace_root
    
    try:
        # 1. Path Validation & Security
        root = Path(workspace_root).resolve()
        target_path = (root / filename).resolve()

        if not str(target_path).startswith(str(root)):
            raise PermissionError(f"Access Denied: {filename} is outside the sandbox.")

        # 2. Existence Check
        if not target_path.exists():
            raise FileNotFoundError(f"Cannot update '{filename}': File does not exist.")

        # 3. Preservation of State (Optional but recommended: Backups)
        # In a high-stakes app, you'd copy to .bak here.

        # 4. Atomic Write / Append
        mode = "a" if append else "w"
        with open(target_path, mode, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # 5. Feedback Loop
        action_verb = "Appended to" if append else "Overwrote"
        result = ToolResult(
            success=True,
            output={
                "path": str(target_path), 
                "operation": "append" if append else "write",
                "new_size": target_path.stat().st_size
            },
            meaning=f"Successfully {action_verb.lower()} {filename}."
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Failed to update file '{filename}': {str(e)}"
        )

    return result.to_state_update()   
@tool
def delete_file(filename: str, runtime_config: ToolRuntime[ContextSchema]) -> dict:
    """
    Standardizes how an agent deletes files from the workspace, with built-in guardrails.
    Args:
        filename: Relative path to the file within the agent's allowed directory.
        runtime_config: The configuration for the tool execution, which may include:
            - workspace_root: The root directory for the agent's file operations.
    Returns:
        A structured response containing:
        - success: Whether the delete operation was successful.
        - output: If successful, includes the path of the deleted file.
        - error: If failed, a detailed error message.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.
        
    """
    workspace_root = runtime_config.context.tool_config.workspace_root
    try:
        # 1. Path Validation & Security
        root = Path(workspace_root).resolve()
        target_path = (root / filename).resolve()

        # Prevent deleting the workspace root or escaping it
        if target_path == root:
            raise PermissionError("Cannot delete the workspace root directory.")
        
        if not str(target_path).startswith(str(root)):
            raise PermissionError(f"Access Denied: {filename} is outside the sandbox.")

        # 2. Existence Check
        if not target_path.exists():
            # We return success=False because the agent's mental model 
            # of the file existing was wrong.
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        # 3. Execution (The actual deletion)
        # Check if it's a file or directory to prevent errors
        if target_path.is_dir():
            # For safety, we might restrict agents to only deleting files.
            # If you want folder deletion, use shutil.rmtree(target_path)
            raise IsADirectoryError(f"'{filename}' is a directory. This tool only deletes files.")
        
        target_path.unlink()

        # 5. Feedback Loop & State Sync
        result = ToolResult(
            success=True,
            output={"path": str(target_path), "action": "deleted"},
            meaning=f"Successfully deleted {filename} from the workspace."
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Failed to delete file '{filename}': {str(e)}"
        )

    return result.to_state_update()
@tool
def patch_file(
    filename: str, 
    search_string: str, 
    replace_string: str, 
    runtime_config: ToolRuntime[ContextSchema]
) -> dict:
    """
    Standardizes how an agent performs targeted text replacements in files, with built-in guardrails.
    Args:
        filename: Relative path to the file within the agent's allowed directory.
        search_string: The exact string to find in the file that needs to be replaced.
        replace_string: The string that will replace the search_string in the file.
        runtime_config: The configuration for the tool execution, which may include:
            - workspace_root: The root directory for the agent's file operations.
    Returns:
        A structured response containing:
        - success: Whether the patch operation was successful.
        - output: If successful, includes file path and a summary of changes.
        - error: If failed, a detailed error message.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.

    """
    workspace_root = runtime_config.context.tool_config.workspace_root
    try:
        # 1. Path Validation
        root = Path(workspace_root).resolve()
        target_path = (root / filename).resolve()
        if not str(target_path).startswith(str(root)):
            raise PermissionError("Path traversal blocked.")

        if not target_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found.")

        # 2. Read and Validate Search String
        content = target_path.read_text(encoding="utf-8")
        
        if search_string not in content:
            # We provide feedback so the agent can try a different search string
            raise ValueError(f"Could not find exact match for search_string in {filename}.")
        
        if content.count(search_string) > 1:
            raise ValueError(f"Multiple occurrences of search_string found. Please be more specific.")

        # 3. Perform the Swap
        new_content = content.replace(search_string, replace_string)

        # 4. Atomic Write
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())

        result = ToolResult(
            success=True,
            output={"path": filename, "changes": "1 block replaced"},
            meaning=f"Successfully patched {filename}."
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Patch failed: {str(e)}"
        )

    return result.to_state_update()
@tool
def execute_file(
    filename: str, 
    cmd_args: List[str], # Changed from 'args' to avoid LangChain internal naming collisions
    runtime_config: ToolRuntime[ContextSchema]
) -> dict:
    """
    Executes a Python script located in the agent's workspace, with built-in guardrails.
    Args:
        filename: Relative path to the Python script within the agent's allowed directory.
        cmd_args: list of string arguments to pass to the script (e.g. ["--mode", "train", "data.csv"]). Provide [] for empty args
        runtime_config: The configuration for the tool execution, which may include:
            - workspace_root: The root directory for the agent's file operations.
            - execution_timeout: Maximum time in seconds to allow the script to run (to prevent infinite loops).
    Returns:
        A structured response containing:
        - success: Whether the execution was successful.
        - output: If successful, includes stdout and return code.
        - error: If failed, a detailed error message or stderr output.
        - meaning: A human-readable summary for the agent's context, guiding its next steps.

    """
    workspace_root = runtime_config.context.tool_config.workspace_root
    timeout = int(runtime_config.context.tool_config.execution_timeout)
    
    try:
        # 1. Path Validation
        root = Path(workspace_root).resolve()
        target_path = (root / filename).resolve()
        if not str(target_path).startswith(str(root)):
            raise PermissionError("Execution outside sandbox blocked.")
        if not target_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found.")
            
        # 2. Build Subprocess Command List
        # Base command: [python, script_path]
        cmd = [sys.executable, str(target_path)]
        
        # Safely append arguments if the agent provided any
        if cmd_args or len(cmd_args) > 0:
            # Enforce that all items are strings to prevent subprocess typing crashes
            cmd.extend([str(a) for a in cmd_args])

        # 3. Subprocess Execution
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(root)  # Ensures the script can resolve workspace-relative imports
        )
        
        # 4. Handle Results
        if process.returncode == 0:
            result = ToolResult(
                success=True,
                output={"stdout": process.stdout, "return_code": 0},
                meaning=f"Execution successful:\n{process.stdout}"
            )
        else:
            # The script ran but crashed (e.g. SyntaxError, RuntimeError)
            result = ToolResult(
                success=False,
                output={"stdout": process.stdout},
                error=process.stderr,
                meaning=f"Execution failed with code {process.returncode}:\n{process.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        result = ToolResult(
            success=False,
            output=None,
            error="Timeout",
            meaning=f"Execution timed out after {timeout} seconds."
        )
    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"System error during execution: {str(e)}"
        )
        
    return result.to_state_update()

@tool
def get_directory_contents(
    target_dir: str,
    recursive: bool,
    runtime_config: ToolRuntime[ContextSchema]
) -> dict:
    """
    Lists the contents of a directory in the agent's workspace, with built-in guardrails.
    Args:
    target_dir: Relative path to the directory within the agent's allowed workspace.
    recursive: If True, lists all nested files and folders; if False, only the immediate contents.
    runtime_config: The configuration for the tool execution, which may include:
    - workspace_root: The root directory for the agent's file operations.
    Returns:
    A structured response containing:
    - success: Whether the operation was successful.
    - output: If successful, includes a list of items with their type (file/folder), size, and last modified time.
    - error: If failed, a detailed error message.
    - meaning: A human-readable summary for the agent's context, guiding its next steps.
    
    """
    workspace_root = runtime_config.context.tool_config.workspace_root
    try:
        # 1. Path Validation & Security
        root = Path(workspace_root).resolve()
        # Ensure we are looking inside the root
        current_path = (root / target_dir).resolve()

        if not str(current_path).startswith(str(root)):
            raise PermissionError(f"Access Denied: Path '{target_dir}' is outside the sandbox.")

        if not current_path.exists():
            raise FileNotFoundError(f"The directory '{target_dir}' does not exist.")

        # 2. Discovery Logic
        items = []
        # Use rglob for recursion or iterdir for flat list
        search_path = current_path.rglob("*") if recursive else current_path.iterdir()

        for path in search_path:
            # Skip hidden files/folders (optional, but keeps agent focused)
            if any(part.startswith('.') for part in path.relative_to(root).parts):
                continue
                
            stats = path.stat()
            items.append({
                "name": path.name,
                "relative_path": str(path.relative_to(root)),
                "type": "directory" if path.is_dir() else "file",
                "size_bytes": stats.st_size,
                "last_modified": stats.st_mtime
            })

        # 3. Semantic Feedback
        count = len(items)
        meaning = f"I found {count} items in '{target_dir}'."
        if recursive:
            meaning = f"Recursively mapped {count} items starting from '{target_dir}'."

        result = ToolResult(
            success=True,
            output={"items": items, "base_path": target_dir},
            meaning=meaning
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Failed to list directory '{target_dir}': {str(e)}"
        )

    return result.to_state_update()
@tool
def get_pwd_context(
    runtime_config: ToolRuntime[ContextSchema]
) -> dict:
    """
    Returns the current working directory context relative to the sandbox root.
    """
    workspace_root= runtime_config.context.tool_config.workspace_root
    current_relative_path = runtime_config.context.tool_config.current_relative_path
    try:
        # 1. Resolve Paths
        root = Path(workspace_root).resolve()
        # The agent might think it's in a subfolder; we track that via 'current_relative_path'
        active_dir = (root / current_relative_path).resolve()

        # 2. Security Check (Sandbox Boundary)
        if not str(active_dir).startswith(str(root)):
            active_dir = root # Force back to root if it tried to escape
            current_relative_path = "."

        # 3. Build Breadcrumbs (The Semantic Map)
        # This helps the LLM understand the hierarchy it traveled through
        relative_to_root = active_dir.relative_to(root)
        breadcrumbs = ["root"] + list(relative_to_root.parts)

        # 4. Feedback Loop
        output = {
            "absolute_path": str(active_dir),
            "relative_path": str(relative_to_root) if str(relative_to_root) != "." else "/",
            "breadcrumbs": " > ".join(breadcrumbs),
            "is_root": active_dir == root
        }

        result = ToolResult(
            success=True,
            output=output,
            meaning=f"You are currently in: {output['relative_path']} (Breadcrumbs: {output['breadcrumbs']})"
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Failed to retrieve directory context: {str(e)}"
        )

    return result.to_state_update()

# Assuming these are imported from your agent.models
# from agent.models import ToolRuntime, ContextSchema, ToolResult

@tool
def search_internet(
    query: str,
    runtime_config: ToolRuntime[ContextSchema]
) -> dict:
    """
    Searches the internet to find real-time information, news, or technical documentation.
    Use this when the user asks about current events or topics outside your local workspace.

    Args:
        query: The specific search string or question to look up.
    """
    # Tavily is pre-optimized  for AI agents
    search = TavilySearch(
        max_results=5,
        search_depth="advanced", # "advanced" is better for technical/detailed queries
        include_answer=True,      # Returns a short AI-generated summary of the results
        include_raw_content=False # Keep it clean to save tokens
    )

    try:
        # 1. Execute the search
        raw_results = search.invoke({"query": query})
        
        # 2. Format findings for the agent's mental model
        # Tavily returns a list of dicts with 'url' and 'content'
        formatted_output = {
            "query": query,
            "results": raw_results
        }

        # 3. Create semantic meaning
        # If Tavily generated an answer, we use that as the primary 'meaning'
        result = ToolResult(
            success=True,
            output=formatted_output,
            meaning=f"Internet search for '{query}' completed. Found {len(raw_results)} relevant sources."
        )

    except Exception as e:
        result = ToolResult(
            success=False,
            output=None,
            error=str(e),
            meaning=f"Internet search failed for '{query}': {str(e)}"
        )

    return result.to_state_update()
ALL_TOOLS = [ 
                get_directory_contents,
                get_pwd_context,
                create_file,
                read_file,
                update_file,
                patch_file,
                delete_file,  
                execute_file,
                search_internet,
                search_memory
            ]
