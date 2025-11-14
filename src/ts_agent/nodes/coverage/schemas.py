"""
Schemas for the Coverage node.
"""

from typing import Dict, Any, List, Optional, TypedDict, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from typing import Any


class DataGap(BaseModel):
    """Schema for identifying missing data."""
    model_config = {"extra": "forbid"}
    
    gap_type: str = Field(..., description="Type of missing data (e.g., 'user_profile', 'application_details')")
    description: str = Field(..., description="Description of what data is missing")


class CoverageRequest(BaseModel):
    """Schema for coverage node input."""
    conversation_history: List[Dict[str, Any]] = Field(..., description="Full conversation history")
    tool_results: List[Dict[str, Any]] = Field(..., description="Results from executed tools")
    successful_tools: List[str] = Field(..., description="Names of successfully executed tools")
    failed_tools: List[str] = Field(..., description="Names of failed tools")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ActionDecision(BaseModel):
    """Schema for coverage's decision to execute an action tool."""
    model_config = {"extra": "forbid"}
    
    action_tool_name: str = Field(..., description="Name of the action tool to execute")
    reasoning: str = Field(..., description="Why Coverage decided to execute this action tool now")
    parameters: Dict[str, Any] = Field(..., description="Complete parameters for the action tool based on the tool schema and gathered data")


class CoverageResponse(BaseModel):
    """Schema for coverage node output."""
    model_config = {"extra": "forbid"}
    
    data_sufficient: bool = Field(..., description="Whether we have sufficient data to respond")
    missing_data: List[DataGap] = Field(default_factory=list, description="List of missing data gaps (empty if data is sufficient)")
    reasoning: str = Field(..., description="Detailed reasoning for the coverage assessment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis (0.0-1.0)")
    next_action: str = Field(..., description="Next action: 'continue' (sufficient data), 'gather_more' (need more data), 'execute_action' (run action tool), 'escalate' (cannot gather data)")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation (required if next_action is 'escalate')")
    action_decision: Optional[ActionDecision] = Field(None, description="Decision to execute an action tool from Plan's suggestions (required if next_action is 'execute_action')")


class CoverageData(TypedDict, total=False):
    """Data structure for coverage node (stored in hop)."""
    coverage_response: CoverageResponse  # Complete analysis from LLM
    next_node: str  # Routing decision: "plan", "respond", "action", "escalate", "end"
