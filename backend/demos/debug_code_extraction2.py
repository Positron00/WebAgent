#!/usr/bin/env python
"""
Simple script to check if our file saving functionality works correctly.
"""

import os
import sys
import re
import json
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_file_saving")

def save_code_to_file(code, filename):
    """Save code to a file in the demo_outputs directory."""
    # Use the absolute path from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "demo_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w") as f:
            f.write("# Test code file\n\n")
            f.write(code)
        
        logger.info(f"Code saved to: {filepath}")
        print(f"Code saved to: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving to file: {e}")
        print(f"Error saving to file: {e}")
        return False

def main():
    """Test saving files to the demo_outputs directory."""
    logger.info("Testing file saving functionality")
    
    # Test saving a simple file
    sample_code = """
def hello_world():
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()
"""
    
    success = save_code_to_file(sample_code, "test_file_saving.py")
    
    if success:
        logger.info("File saving test successful")
    else:
        logger.error("File saving test failed")

if __name__ == "__main__":
    main() 