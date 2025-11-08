"""
Schemas for the Plan node.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field, validator


class ToolCall(BaseModel):
    """Schema for a single tool call in the plan."""
    model_config = {"extra": "ignore"}  # Silently drop extra fields for robustness
    
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Dictionary containing ALL tool-specific parameters (e.g., user_email, query, threshold). Do NOT put parameters at top level."
    )
    reasoning: str = Field(..., description="Why this tool call is needed")


class PlanData(TypedDict, total=False):
    """Data structure for plan node (stored in hop)."""
    plan: Optional[Dict[str, Any]]
    tool_calls: Optional[List[ToolCall]]  # All tool calls (for backward compatibility)
    gather_tool_calls: Optional[List[ToolCall]]  # Only gather-type tools
    action_tool_calls: Optional[List[ToolCall]]  # Only action-type tools
    reasoning: Optional[str]


class Plan(BaseModel):
    """Schema for the agent's execution plan with structured tool calls."""
    model_config = {"extra": "ignore"}  # Silently drop extra fields for robustness
    
    reasoning: str = Field(..., description="Why this plan was created and overall strategy")
    tool_calls: List[ToolCall] = Field(..., description="List of tool calls to execute in order")


class PlanRequest(BaseModel):
    """Schema for plan node input."""
    conversation_history: List[Dict[str, Any]] = Field(..., description="Full conversation history")
    user_email: Optional[str] = Field(None, description="User email if available")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context from previous hops")
