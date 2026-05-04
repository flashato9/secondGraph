import os

def list_image_files(directory='.'):
    """Lists all image files in the given directory."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    return [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

def get_image_bytes(image_path):
    """Reads an image file and returns its raw bytes."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, 'rb') as f:
        return f.read()
