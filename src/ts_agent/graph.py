"""LangGraph for the agent."""

from langgraph.graph import StateGraph, START, END
from ts_agent.types import State
from ts_agent.nodes.initialize.initialize import initialize_node
from ts_agent.nodes.procedure.procedure import procedure_node
from ts_agent.nodes.plan.plan import plan_node
from ts_agent.nodes.gather.gather import gather_node
from ts_agent.nodes.coverage.coverage import coverage_node
from ts_agent.nodes.action.action import action_node
from ts_agent.nodes.draft.draft import draft_node
from ts_agent.nodes.validate.validate import validate_node
from ts_agent.nodes.escalate.escalate import escalate_node
from ts_agent.nodes.response.response import response_node
from ts_agent.nodes.finalize.finalize import finalize_node


def build_graph():
    """Build the agent graph."""
    g = StateGraph(State)
    
    # Add nodes
    g.add_node("initialize", initialize_node)
    g.add_node("procedure", procedure_node)
    g.add_node("plan", plan_node)
    g.add_node("gather", gather_node)
    g.add_node("coverage", coverage_node)
    g.add_node("action", action_node)
    g.add_node("draft", draft_node)
    g.add_node("validate", validate_node)
    g.add_node("escalate", escalate_node)
    g.add_node("response", response_node)
    g.add_node("finalize", finalize_node)
    
    # Add edges
    g.add_edge(START, "initialize")
    
    # Add conditional routing from plan
    def route_from_plan(state: State) -> str:
        """Route from plan node based on success/failure."""
        if state.get("error"):
            return "escalate"  # If plan generation failed, escalate
        else:
            return "gather"  # If successful, proceed to gather
    
    # Add conditional routing from gather
    def route_from_gather(state: State) -> str:
        """Route from gather node based on success/failure."""
        if state.get("error"):
            return "escalate"  # If gather failed, escalate
        else:
            return "coverage"  # If successful, proceed to coverage
    
    # Add conditional routing from coverage
    def route_from_coverage(state: State) -> str:
        """Route from coverage node based on analysis."""
        next_node = state.get("next_node", "end")
        
        if next_node == "plan":
            return "plan"
        elif next_node == "respond":
            return "draft"  # Route to draft node for response generation
        elif next_node == "action":
            return "action"  # Route to action node for action tool execution
        elif next_node == "escalate":
            return "escalate"  # Route to escalate node
        else:
            return "end"
    
    # Add conditional routing from initialize
    def route_from_initialize(state: State) -> str:
        """Route from initialize node based on success/failure."""
        if state.get("error"):
            return "escalate"  # If initialization failed, escalate
        else:
            return "procedure"  # If successful, proceed to procedure
    
    # Add conditional routing from validate
    def route_from_validate(state: State) -> str:
        """Route from validate node based on validation result."""
        next_node = state.get("next_node", "end")
        
        if next_node == "response":
            return "response"  # Validation passed, send response
        elif next_node == "draft":
            return "draft"  # Validation failed on first attempt, retry draft
        elif next_node == "escalate":
            return "escalate"  # Validation failed on second attempt, escalate
        else:
            return "end"
    
    # Add conditional routing from draft
    def route_from_draft(state: State) -> str:
        """Route from draft node based on response type or error."""
        next_node = state.get("next_node")
        if next_node == "escalate":
            return "escalate"  # ROUTE_TO_TEAM or error
        else:
            return "validate"  # Normal REPLY, proceed to validation
    
    g.add_conditional_edges(
        "initialize",
        route_from_initialize,
        {
            "procedure": "procedure",
            "escalate": "escalate"
        }
    )
    
    g.add_edge("procedure", "plan")
    
    g.add_conditional_edges(
        "plan",
        route_from_plan,
        {
            "gather": "gather",
            "escalate": "escalate"
        }
    )
    
    g.add_conditional_edges(
        "gather",
        route_from_gather,
        {
            "coverage": "coverage",
            "escalate": "escalate"
        }
    )
    
    g.add_conditional_edges(
        "coverage",
        route_from_coverage,
        {
            "plan": "plan",
            "draft": "draft",
            "action": "action",
            "escalate": "escalate",
            "end": END
        }
    )
    
    # Add conditional routing from action
    def route_from_action(state: State) -> str:
        """Route from action node back to coverage for re-evaluation."""
        next_node = state.get("next_node", "coverage")
        
        if next_node == "coverage":
            return "coverage"
        elif next_node == "escalate":
            return "escalate"
        else:
            return "coverage"  # Default to coverage
    
    g.add_conditional_edges(
        "action",
        route_from_action,
        {
            "coverage": "coverage",
            "escalate": "escalate"
        }
    )
    
    # Draft node routes conditionally
    g.add_conditional_edges(
        "draft",
        route_from_draft,
        {
            "validate": "validate",
            "escalate": "escalate"
        }
    )
    
    # Validate node routes conditionally
    g.add_conditional_edges(
        "validate",
        route_from_validate,
        {
            "response": "response",
            "draft": "draft",
            "escalate": "escalate",
            "end": END
        }
    )
    
    # Add conditional routing from response
    def route_from_response(state: State) -> str:
        """Route from response node based on delivery success/failure and next_node."""
        next_node = state.get("next_node", "finalize")
        
        # Response node sets next_node to "escalate" when:
        # - Response delivery failed, OR
        # - Draft requested escalation (ROUTE_TO_TEAM), OR
        # - Actions were taken (needs human review)
        if next_node == "escalate":
            return "escalate"
        else:
            return "finalize"
    
    g.add_conditional_edges(
        "response",
        route_from_response,
        {
            "finalize": "finalize",
            "escalate": "escalate"
        }
    )
    
    # Escalate node goes to finalize
    g.add_edge("escalate", "finalize")
    
    # Finalize node always ends
    g.add_edge("finalize", END)
    
    return g.compile()


# Export the graph for LangGraph CLI
graph = build_graph()