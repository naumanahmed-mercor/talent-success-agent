"""
Debug utilities for development and testing.
"""

import os
import json
import time
from typing import Dict, Any, Optional


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled via DEBUG_PROMPTS env var."""
    return os.getenv("DEBUG_PROMPTS", "false").lower() in ("true", "1", "yes")


def dump_prompt_to_file(
    prompt: str,
    node_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    suffix: str = ""
) -> Optional[str]:
    """
    Dump a prompt to a debug file if DEBUG_PROMPTS is enabled.
    
    Args:
        prompt: The prompt text to dump
        node_name: Name of the node (e.g., "plan", "draft", "procedure")
        metadata: Optional metadata dict to include in the file
        suffix: Optional suffix for the filename (e.g., "_retry")
        
    Returns:
        Path to the dumped file if debug is enabled, None otherwise
    """
    if not is_debug_enabled():
        return None
    
    debug_dir = os.path.join(os.getcwd(), "debug_prompts")
    os.makedirs(debug_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{node_name}_prompt_{timestamp}{suffix}.txt"
    filepath = os.path.join(debug_dir, filename)
    
    with open(filepath, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"{node_name.upper()} NODE - FULL PROMPT\n")
        f.write("="*80 + "\n\n")
        f.write(prompt)
        
        if metadata:
            f.write("\n\n" + "="*80 + "\n")
            f.write("METADATA\n")
            f.write("="*80 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    print(f"📝 DEBUG: {node_name.capitalize()} prompt saved to {filepath}")
    return filepath


def dump_response_to_file(
    response_data: Dict[str, Any],
    node_name: str,
    suffix: str = ""
) -> Optional[str]:
    """
    Dump a structured response to a debug JSON file if DEBUG_PROMPTS is enabled.
    
    Args:
        response_data: The response data to dump (will be JSON serialized)
        node_name: Name of the node (e.g., "plan", "draft")
        suffix: Optional suffix for the filename (e.g., "_retry")
        
    Returns:
        Path to the dumped file if debug is enabled, None otherwise
    """
    if not is_debug_enabled():
        return None
    
    debug_dir = os.path.join(os.getcwd(), "debug_prompts")
    os.makedirs(debug_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{node_name}_response_{timestamp}{suffix}.json"
    filepath = os.path.join(debug_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(response_data, f, indent=2)
    
    print(f"📝 DEBUG: {node_name.capitalize()} response saved to {filepath}")
    return filepath

