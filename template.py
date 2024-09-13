import os
from pathlib import Path
import logging #for tracking events, debugging, and understanding the application's flow


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s')

#creating a list of file

file_list = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/chat.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html",
    "test.py"


]

for filepath in file_list:
    filepath = Path(filepath)  
    filedir, filename = filepath.parent, filepath.name  # Extracting directory and filename

    # Creating the directory if it does not exist
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created the directory: {filedir} for file {filename}")
    
    # Create the file if it doesn't exist or if it's empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w') as f:
            pass  # Creates an empty file
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
