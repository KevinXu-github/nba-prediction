import os
import json
from datetime import datetime

def save_to_json(data, directory, filename):
    """Save data to a JSON file"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    return filepath

def load_from_json(filepath):
    """Load data from a JSON file"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)