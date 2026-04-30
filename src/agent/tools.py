


import base64
from pathlib import Path
from typing import Annotated

import aiofiles
from aiofiles import os
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage
import yaml

# Use a consistent path for your "database"
DB_PATH = Path("static/image_db.yml")

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
    file_extension = Path(path).suffix.replace(".","")
    image64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = f"image/{file_extension}"
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
                    "url": f"data:{mime_type};base64,{image64}"
                },
        },
        
    ]
    return content

async def upload_latest_image(file_name:str,image_description:str, state: Annotated[dict, InjectedState]) -> str:
    """
    Uploads an image to the server and saves its description.
    Args:
    - file_name: The name to save the file as (without extension).
    - image_description: A description of the image to be saved in the database.
    - state: The current graph state, injected automatically.
    Returns:
    - A success message with the file path.
    """
    
    # 1. Look through messages in reverse to find the last image
    messages = state.get("messages", [])
    image_data_base64 = None
    mime_type = "image/jpeg" # Default

    for msg in reversed(messages):
        # We check HumanMessages specifically as they contain the uploads
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    # 1. Check for your specific debug structure
                    if part.get("type") == "image" and "data" in part:
                        image_data_base64 = part["data"]
                        break
                    # 2. Fallback check for standard image_url format
                    elif "image_url" in part:
                        url = part["image_url"]["url"]
                        image_data_base64 = url.split(",")[-1]
                        break
        if image_data_base64:
            break
    if not image_data_base64:
        raise Exception("Error: I couldn't find an image in our recent history to save. Please upload one!")
    image_data = None
    try:
        image_data = base64.b64decode(image_data_base64)
    except Exception as e:
        raise Exception(f"Error decoding image data: {str(e)}")
    
    save_path = Path("static/uploads") / f"{file_name}.jpg"
    # CRITICAL CHECK: Throw exception if file exists
    if save_path.exists():
        raise Exception(f"Conflict: A file named '{file_name}.jpg' already exists. Please choose a different name.")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(image_data)
    await _save_image_description(str(save_path), image_description)
    return f"File saved successfully at {save_path}"

def get_image_database() -> list[dict]:
    """
    Retrieves the image database from a YAML file. Each entry contains the image path, description, and timestamp.
    Returns:
    - A list of dictionaries, each representing an image entry with keys: 'path', 'description', and 'timestamp'.
    """
    
    import yaml
    #note to self: this is an example of how you can create a tool that accesses a database. In this case we are using a yaml file as a simple database to save image paths and descriptions. The get_image_database tool reads from this file and returns the data to the model. You can create more complex tools that access more complex databases as needed.
    if DB_PATH.exists():
        with open(DB_PATH, "r") as f:
            data = yaml.safe_load(f) or []
            return data
    else:
        return []
    
async def delete_image(file_path: str) -> str:
    """
    Deletes an image file and its corresponding database entry.
    Args:
    - file_path: The path of the image file to be deleted.
    Returns:
    - A message indicating the result of the deletion process.
    """
    # Helper 1: Normalize path
    target = _resolve_path(file_path)
    
    # Method 1: Delete File
    file_result = await _remove_file_from_disk(target)
    
    # Method 2: Delete Metadata
    meta_result = await _remove_metadata_entry(target)
    
    return f"{file_result} | {meta_result}"

def _resolve_path(path_str: str) -> Path:
    """Helper: Standardizes paths to ensure comparisons always work."""
    if not path_str:
        return Path("null")
    return Path(path_str).resolve()

def _load_yaml_db() -> list:
    """Helper: Safe loading for the YAML file."""
    with open(DB_PATH, "r") as f:
        return yaml.safe_load(f) or []

def _save_yaml_db(data: list):
    """Helper: Standardized saving for the YAML file."""
    with open(DB_PATH, "w") as f:
        yaml.dump(data, f, sort_keys=False)
        
async def _remove_file_from_disk(resolved_path: Path) -> str:
    """Method: Handles the physical deletion of the image."""
    try:
        if resolved_path.exists():
            await aiofiles.os.remove(resolved_path)
            return "File deleted from disk"
        return "File not found on disk"
    except Exception as e:
        return f"File deletion failed: {str(e)}"

async def _remove_metadata_entry(resolved_path: Path) -> str:
    """Method: Handles the YAML database update."""
    try:
        if not DB_PATH.exists():
            return "DB not found"
            
        # Helper 2: Read/Write YAML logic
        data = _load_yaml_db()
        
        # Filter logic
        updated_data = [
            item for item in data 
            if _resolve_path(item.get("path")) != resolved_path
        ]
        
        _save_yaml_db(updated_data)
        return "Metadata removed"
    except Exception as e:
        return f"Metadata removal failed: {str(e)}"

async def _save_image_description(image_file_path: str, image_description:str) -> str:
    import yaml
    import os
    from pathlib import Path

    """
    Saves image metadata to a YAML file. 
    Returns the full list of current entries.
    """
    # 1. Load existing data if the file exists
    if DB_PATH.exists():
        with open(DB_PATH, "r") as f:
            # Safe load handles the conversion from YAML to Python list
            data = yaml.safe_load(f) or []
    else:
        data = []

    # 2. Prepare the new entry
    new_entry = {
        "path": image_file_path,
        "description": image_description,
        "timestamp": os.path.getmtime(image_file_path) if os.path.exists(image_file_path) else None
    }

    # 3. Append and save back to the file
    data.append(new_entry)
    
    with open(DB_PATH, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    return "Image description saved successfully."
