"""
Shared utility for sanitizing tool parameters.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from ts_agent.types import ToolType

logger = logging.getLogger(__name__)


def sanitize_tool_params(
    params: Dict[str, Any],
    input_schema: Dict[str, Any],
    tool_name: str,
    injection_map: Dict[str, Any],
    tool_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sanitize tool parameters by injecting verified values from state.
    
    This function:
    1. Accepts a dict of trusted values to inject/replace
    2. Goes through tool schema and replaces params that match the injection map
    3. Validates all required params are present after injection
    4. For action tools, automatically injects conversation_id and dry_run
    
    Args:
        params: Original parameters from LLM
        input_schema: Tool's input schema
        tool_name: Name of the tool
        injection_map: Dict mapping param names to trusted values (can be callables)
        tool_type: Type of tool (ToolType.GATHER, ToolType.INTERNAL_ACTION, etc.)
                   If provided and is an action type, conversation_id and dry_run will be auto-injected
        
    Returns:
        Sanitized parameters with trusted values injected
        
    Raises:
        ValueError: If required parameters are missing after injection
    """
    sanitized = params.copy()
    properties = input_schema.get("properties", {})
    required_params = input_schema.get("required", [])
    
    # For action tools, ensure conversation_id and dry_run are in injection map
    is_action_tool = tool_type in [ToolType.INTERNAL_ACTION.value, ToolType.EXTERNAL_ACTION.value]
    if is_action_tool:
        # Auto-add conversation_id and dry_run to injection map if not present
        if "conversation_id" in properties and "conversation_id" not in injection_map:
            # conversation_id should have been provided in injection_map for action tools
            logger.warning(f"conversation_id not in injection_map for action tool {tool_name}")
        
        if "dry_run" in properties and "dry_run" not in injection_map:
            injection_map["dry_run"] = lambda: os.getenv("DRY_RUN", "false").lower() == "true"
    
    # Go through each parameter in the tool's schema
    for param_name in properties.keys():
        # Check if this param should be injected/replaced
        if param_name in injection_map:
            value = injection_map[param_name]
            # Handle callable values (like dry_run)
            injected_value = value() if callable(value) else value
            
            # Ensure conversation_id is always a string
            if param_name == "conversation_id" and injected_value is not None:
                injected_value = str(injected_value)
            
            sanitized[param_name] = injected_value
            
            if param_name not in params or params[param_name] != sanitized[param_name]:
                logger.info(
                    f"💉 Injected {param_name}={sanitized[param_name]} "
                    f"(was: {params.get(param_name, 'missing')})"
                )
    
    # Validate all required parameters are present
    missing_params = []
    for required_param in required_params:
        if required_param not in sanitized or sanitized[required_param] is None:
            missing_params.append(required_param)
    
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {tool_name}: {', '.join(missing_params)}"
        )
    
    return sanitized


