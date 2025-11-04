"""
Schemas for the Validate node.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ValidationResponse(BaseModel):
    """
    Simplified response from validation endpoint.
    We only care about overall_passed for routing logic.
    """
    overall_passed: bool = Field(..., description="Whether validation passed")
    
    class Config:
        extra = "allow"  # Allow additional fields, ignore them


class ValidateData(BaseModel):
    """Data structure for validate node (stored at state level)."""
    validation_response: Optional[Dict[str, Any]] = Field(None, description="Raw validation response")
    overall_passed: bool = Field(..., description="Whether validation passed")
    validation_note_added: bool = Field(False, description="Whether note was added to Intercom")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation if validation failed")
    next_action: str = Field(..., description="Next action: 'response', 'draft', or 'escalate'")
    retry_count: int = Field(0, description="Number of times validation has been retried")
