"""
Schemas for the Finalize node.
"""

from typing import Optional
from pydantic import BaseModel, Field


class FinalizeData(BaseModel):
    """Data structure for finalize node (stored at state level)."""
    melvin_status: str = Field(..., description="Melvin Status value set in Intercom")
    status_updated: bool = Field(False, description="Whether Melvin Status was updated")
    conversation_snoozed: bool = Field(False, description="Whether conversation was snoozed")
    snooze_duration_seconds: int = Field(300, description="Snooze duration in seconds (default: 5 minutes)")
    error: Optional[str] = Field(None, description="Error message if finalization failed")
    webhook_posted: Optional[bool] = Field(None, description="Whether response was posted to webhook (test mode only)")
