"""Types and state definitions for the agent."""

from enum import Enum
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from typing_extensions import TypedDict


class ToolType(str, Enum):
    """Types of tools available."""
    GATHER = "gather"
    ACTION = "action"

if TYPE_CHECKING:
    from ts_agent.nodes.plan.schemas import PlanData
    from ts_agent.nodes.gather.schemas import GatherData
    from ts_agent.nodes.coverage.schemas import CoverageData


class HopData(TypedDict, total=False):
    """Data for a single hop with nested structure."""
    hop_number: int
    plan: Optional[Dict[str, Any]]
    gather: Optional[Dict[str, Any]]
    coverage: Optional[Dict[str, Any]]


class Message(TypedDict, total=False):
    """Individual message in a conversation."""
    role: str  # "user" or "assistant"
    content: str


class UserDetails(TypedDict, total=False):
    """User details for personalization."""
    name: Optional[str]  # User's name (fetched from Intercom)
    email: Optional[str]  # User's email (fetched from Intercom)


class ConversationHistory(TypedDict, total=False):
    """Conversation history with metadata."""
    messages: List[Message]  # Array of conversation messages
    subject: Optional[str]  # Conversation subject/title (often empty)


class State(TypedDict, total=False):
    """State for the LangGraph."""
    # Input (Intercom)
    conversation_id: str  # Intercom conversation ID (primary input)
    messages: List[Message]  # Array of conversation messages (fetched from Intercom)
    user_details: UserDetails  # User details (name and email from Intercom)
    subject: Optional[str]  # Conversation subject/title (fetched from Intercom, often empty)
    melvin_admin_id: Optional[str]  # Melvin bot admin ID for Intercom actions
    timestamp: Optional[str]
    
    # Test Mode Configuration
    mode: Optional[str]  # Execution mode: "test" for dry-run testing
    dry_run: Optional[bool]  # If True, skip all Intercom write operations
    procedure_id: Optional[str]  # Direct procedure ID (bypasses search in test mode)
    
    # MCP Integration
    mcp_client: Optional[Any]  # MCP client instance
    available_tools: Optional[List[Dict[str, Any]]]  # Available tools from MCP server
    
    # Procedure (Retrieved before plan-gather-coverage loop)
    selected_procedure: Optional[Dict[str, Any]]  # Selected procedure from RAG store (if any)
    procedure_node: Optional[Dict[str, Any]]  # Procedure node data (query, results, evaluation)
    
    # Data Storage (Independent of hops)
    tool_data: Optional[Dict[str, Any]]  # Individual tool results by tool name
    docs_data: Optional[Dict[str, Any]]  # Individual docs results by query/topic
    
    # Loop Management (Plan → Gather → Coverage)
    hops: List[HopData]  # Array of hop data
    max_hops: Optional[int]  # Maximum allowed hops (default: 2)
    
    # Action Management (separate from hops)
    actions: Optional[List[Dict[str, Any]]]  # List of action executions with audit trail
    max_actions: Optional[int]  # Maximum allowed actions per flow (default: 1)
    actions_taken: Optional[int]  # Number of actions taken so far
    
    # Validation Management
    max_validation_retries: Optional[int]  # Maximum allowed validation retries (default: 1)
    
    # Post-Loop Nodes (Draft, Validate, Escalate, Response)
    draft: Optional[Dict[str, Any]]  # Draft node data
    validate: Optional[List[Dict[str, Any]]]  # Array of validate node data (one per attempt)
    escalate: Optional[Dict[str, Any]]  # Escalate node data
    response_delivery: Optional[Dict[str, Any]]  # Response node delivery data
    finalize: Optional[Dict[str, Any]]  # Finalize node data
    
    # Routing
    next_node: Optional[str]  # "plan", "respond", "end", "escalate"
    escalation_reason: Optional[str]
    
    # Output
    response: str
    error: Optional[str]
    
    # Intercom Configuration
    metadata: Optional[Dict[str, Any]]  # Additional metadata for Intercom
