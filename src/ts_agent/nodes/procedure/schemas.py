"""
Schemas for the Procedure node.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class QueryGeneration(BaseModel):
    """Schema for generating a procedure query."""
    query: str = Field(..., description="The search query to find relevant procedures")
    reasoning: str = Field(..., description="Why this query will help find relevant procedures")


class ProcedureEvaluation(BaseModel):
    """Schema for evaluating procedure results."""
    is_match: bool = Field(..., description="Whether any procedure perfectly matches the scenario")
    selected_procedure_index: int = Field(-1, description="Index of the selected procedure (0-based), or -1 if none")
    reasoning: str = Field(..., description="Detailed reasoning for selection or rejection")
    selected_procedure_data: Optional[Dict[str, Any]] = Field(None, description="Full procedure data from select endpoint")


class ProcedureResult(BaseModel):
    """A single procedure result from the RAG store."""
    id: Optional[str] = Field(None, description="Procedure ID")
    title: Optional[str] = Field(None, description="Procedure title")
    content: str = Field(..., description="Procedure content")
    relevance_score: Optional[float] = Field(None, description="Relevance score from RAG")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def model_validate(cls, obj):
        """Custom validation to handle integer IDs."""
        if isinstance(obj, dict) and "id" in obj and isinstance(obj["id"], int):
            obj = obj.copy()
            obj["id"] = str(obj["id"])
        return super().model_validate(obj)


class SelectedProcedure(BaseModel):
    """The procedure selected after evaluation."""
    id: Optional[str] = Field(None, description="Procedure ID")
    title: Optional[str] = Field(None, description="Procedure title")
    content: str = Field(..., description="Full procedure content")
    reasoning: str = Field(..., description="Why this procedure was selected")
    relevance_score: Optional[float] = Field(None, description="Relevance score from RAG")


class ProcedureData(BaseModel):
    """Data structure for procedure node (stored at state level)."""
    query: str = Field(..., description="The query generated to search procedures")
    query_reasoning: str = Field(..., description="Reasoning for the query")
    top_k_results: List[ProcedureResult] = Field(default_factory=list, description="Top-k results from RAG")
    selected_procedure: Optional[SelectedProcedure] = Field(None, description="The selected procedure after evaluation")
    evaluation_reasoning: str = Field(..., description="Reasoning for selecting or rejecting procedures")
    timestamp: str = Field(..., description="Procedure node execution timestamp")
    success: bool = Field(..., description="Whether procedure retrieval was successful")
    error: Optional[str] = Field(None, description="Error message if procedure retrieval failed")

