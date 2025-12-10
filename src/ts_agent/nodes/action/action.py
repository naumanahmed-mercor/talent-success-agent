"""Action node implementation for executing action tools."""

import os
import time
from typing import Dict, Any
from ts_agent.types import State
from .schemas import ActionData, ActionResult
from src.mcp.factory import create_mcp_client
from src.clients.intercom import IntercomClient
from src.utils.formatting import format_action_audit_note


def action_node(state: State) -> State:
    """
    Execute an action tool and log audit notes.
    
    This node executes a single action tool (e.g., match_and_link_conversation_to_ticket)
    and records detailed audit notes about the execution since these tools have real-world effects.
    """
    # Get current hop data
    hops_array = state.get("hops", [])
    current_hop_index = len(hops_array) - 1
    
    if current_hop_index < 0:
        state["error"] = "No current hop data found"
        return state
    
    current_hop_data = hops_array[current_hop_index]
    hop_number = current_hop_data.get("hop_number", current_hop_index + 1)
    coverage_data = current_hop_data.get("coverage", {})
    
    # Get the coverage response (contains the action tool call)
    coverage_response = coverage_data.get("coverage_response")
    
    if not coverage_response:
        state["error"] = "No coverage response found"
        return state
    
    # Get the action decision from coverage response
    action_decision = coverage_response.get("action_decision")
    
    if not action_decision:
        state["error"] = "No action decision specified by coverage node"
        return state
    
    # Get action tool name from coverage's decision
    action_tool_name = action_decision.get("action_tool_name")
    coverage_reasoning = action_decision.get("reasoning", "")
    coverage_parameters = action_decision.get("parameters", {})
    
    if not action_tool_name:
        state["error"] = "No action tool name in coverage decision"
        return state
    
    # Coverage now provides the full parameters
    # We just need to inject runtime values like conversation_id
    action_tool_params = coverage_parameters.copy()
    
    # Inject conversation_id at runtime to ensure correctness
    conversation_id = state.get("conversation_id")
    if conversation_id:
        action_tool_params["conversation_id"] = conversation_id
        print(f"💉 Injected conversation_id at runtime: {conversation_id}")
    
    print(f"⚡ Action Node - Executing: {action_tool_name}")
    print("=" * 50)
    
    try:
        # Create MCP client (don't store in state due to serialization issues)
        # Get mode from state for auth token selection
        mode = state.get("mode")
        mcp_client = create_mcp_client(mode=mode)
        
        # Execute the action tool
        start_time = time.time()
        
        print(f"🚀 Executing action tool: {action_tool_name}...")
        # Pass dry_run flag from state to the tool
        dry_run = state.get("dry_run", False)
        result_data = _execute_action_tool(mcp_client, action_tool_name, action_tool_params, dry_run)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Generate formatted audit note
        audit_note = format_action_audit_note(
            action_name=action_tool_name,
            parameters=action_tool_params,
            result=result_data,
            execution_time_ms=execution_time,
            success=True,
            error=None
        )
        
        # Create successful result
        result = ActionResult(
            tool_name=action_tool_name,
            success=True,
            data=result_data,
            execution_time_ms=execution_time,
            timestamp=timestamp,
            audit_notes=audit_note
        )
        
        # Determine if this action requires human review
        # Actions require review if they made actual changes (not just "no matches found")
        requires_review = _action_requires_review(action_tool_name, result_data)
        
        if requires_review:
            print(f"⚠️  Action '{action_tool_name}' requires human review - will escalate after response")
        else:
            print(f"✅ Action '{action_tool_name}' does not require escalation")
        
        # Store action data at state level (not in hop data)
        action_data: ActionData = {
            "tool_name": action_tool_name,
            "tool_result": result.model_dump(),
            "execution_time_ms": execution_time,
            "execution_status": "completed",
            "audit_notes": audit_note,
            "timestamp": timestamp,
            "success": True,
            "error": None,
            "requires_review": requires_review
        }
        
        # Add hop number to track which hop triggered this action
        action_data["hop_number"] = hop_number
        
        # Append to actions list at state level
        if "actions" not in state:
            state["actions"] = []
        state["actions"].append(action_data)
        
        # Increment actions taken counter
        state["actions_taken"] = state.get("actions_taken", 0) + 1
        
        print(f"✅ Action tool executed successfully ({execution_time:.1f}ms)")
        print(f"📊 Actions taken: {state['actions_taken']}/{state.get('max_actions', 1)}")
        
        # Post audit note to Intercom
        _post_action_note_to_intercom(
            state,
            action_tool_name,
            action_tool_params,
            audit_note,
            True,
            None
        )
        
        # Route back to coverage for re-evaluation and response generation
        # The escalation will happen AFTER the response is sent to the user
        state["next_node"] = "coverage"
        print(f"🔄 Routing back to coverage for response generation")
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        error_msg = str(e)
        
        # Generate formatted audit note for failure
        audit_note = format_action_audit_note(
            action_name=action_tool_name,
            parameters=action_tool_params,
            result=None,
            execution_time_ms=execution_time,
            success=False,
            error=error_msg
        )
        
        # Create failed result at state level
        # Failed actions always require review
        action_data: ActionData = {
            "tool_name": action_tool_name,
            "tool_result": None,
            "execution_time_ms": execution_time,
            "execution_status": "failed",
            "audit_notes": audit_note,
            "timestamp": timestamp,
            "success": False,
            "error": error_msg,
            "requires_review": True  # Failed actions always require review
        }
        
        # Add hop number to track which hop triggered this action
        action_data["hop_number"] = hop_number
        
        # Append to actions list at state level
        if "actions" not in state:
            state["actions"] = []
        state["actions"].append(action_data)
        
        # Increment counter even for failed actions
        state["actions_taken"] = state.get("actions_taken", 0) + 1
        
        print(f"❌ Action tool execution failed: {error_msg}")
        
        # Post failure note to Intercom
        _post_action_note_to_intercom(
            state,
            action_tool_name,
            action_tool_params,
            audit_note,
            False,
            error_msg
        )
        
        # For failed actions, escalate immediately without sending a response
        state["next_node"] = "escalate"
        state["escalation_reason"] = f"Action tool failed: {action_tool_name}. Error: {error_msg}. Human review required."
        print(f"🚨 Action failed - escalating immediately")
    
    return state


def _action_requires_review(tool_name: str, result_data: Any) -> bool:
    """
    Determine if an action requires human review based on its result.
    
    Actions require review if they made actual changes, not if they just
    returned "no matches" or similar.
    
    Args:
        tool_name: Name of the action tool
        result_data: Result data from the tool execution
        
    Returns:
        True if the action requires human review, False otherwise
    """
    import json
    
    # Configuration: Tools that should NOT trigger escalation
    # These tools are considered safe or self-contained actions
    NO_ESCALATION_TOOLS = {
        "route_conversation_to_project_client",  # Just routes conversation, no data changes
        "generate_reset_interview_link",  # Generates link for user, no escalation needed
        "generate_reset_tax_document_link",
        "generate_reset_form_link",
    }
    
    # If tool is in the no-escalation list, never require review
    if tool_name in NO_ESCALATION_TOOLS:
        return False
    
    if tool_name == "match_and_link_conversation_to_ticket":
        # Check if the tool actually matched/linked a ticket
        # Result structure: [{"type": "text", "text": "{...JSON...}"}]
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict):
                text = first_item.get("text", "")
                if isinstance(text, str):
                    try:
                        # Parse the JSON string
                        parsed = json.loads(text)
                        # Check the match_found field
                        match_found = parsed.get("match_found", False)
                        # Only require review if a match was found and linked
                        return match_found
                    except json.JSONDecodeError:
                        # If we can't parse, default to requiring review for safety
                        return True
        # Default: require review for safety
        return True
    
    # For other action tools, default to requiring review
    # Add tools to NO_ESCALATION_TOOLS above to skip escalation
    return True


def _execute_action_tool(mcp_client, tool_name: str, parameters: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Execute a single action tool using MCP client with extended timeout.
    
    Action tools may take longer than gather tools due to external API calls,
    so we use a 120-second timeout instead of the default 30 seconds.
    
    Args:
        mcp_client: MCP client instance
        tool_name: Action tool name to execute
        parameters: Parameters for the action tool
        dry_run: If True, pass dry_run parameter to the tool
        
    Returns:
        Action tool execution result data
    """
    # Add dry_run parameter to tool parameters if specified
    tool_parameters = parameters.copy()
    if dry_run:
        tool_parameters["dry_run"] = True
        print(f"🧪 Dry run mode: passing dry_run=True to {tool_name}")
    
    try:
        # Use 120-second timeout for action tools (vs 30s default for gather tools)
        result = mcp_client.call_tool(tool_name, tool_parameters, timeout=120.0)
        return result
    except Exception as e:
        raise Exception(f"Action tool execution failed: {str(e)}")


# Note: _generate_audit_notes has been replaced by format_action_audit_note from src.utils.formatting


def _post_action_note_to_intercom(
    state: State,
    tool_name: str,
    parameters: Dict[str, Any],
    audit_notes: str,
    success: bool,
    error: str = None
) -> None:
    """
    Post action execution audit note to Intercom conversation.
    
    This creates an internal note visible to human agents documenting
    what action was automatically performed.
    
    Args:
        state: Current state with conversation_id and admin_id
        tool_name: Name of the action tool executed
        parameters: Parameters used
        audit_notes: Generated audit notes
        success: Whether execution succeeded
        error: Error message if failed
    """
    try:
        conversation_id = state.get("conversation_id")
        admin_id = state.get("melvin_admin_id")
        
        if not conversation_id or not admin_id:
            print("⚠️  Cannot post action note: missing conversation_id or admin_id")
            return
        
        # Get Intercom API key
        intercom_api_key = os.getenv("INTERCOM_API_KEY")
        if not intercom_api_key:
            print("⚠️  Cannot post action note: INTERCOM_API_KEY not found")
            return
        
        # Initialize Intercom client with dry_run from state
        dry_run = state.get("dry_run", False)
        intercom_client = IntercomClient(intercom_api_key, dry_run=dry_run)
        
        # Build note content
        status_emoji = "✅" if success else "❌"
        status_text = "SUCCESS" if success else "FAILED"
        
        note_lines = [
            f"🤖 **Melvin Action Executed**",
            f"",
            f"{status_emoji} **Status:** {status_text}",
            f"**Action:** `{tool_name}`",
            f"**Parameters:** `{parameters}`",
            f"",
            f"**Audit Trail:**",
            f"{audit_notes}",
        ]
        
        if error:
            note_lines.append(f"")
            note_lines.append(f"**Error:** {error}")
        
        note_content = "\n".join(note_lines)
        
        # Post note to Intercom
        print(f"📝 Posting action audit note to Intercom conversation {conversation_id}")
        intercom_client.add_note(
            conversation_id=conversation_id,
            note_body=note_content,
            admin_id=admin_id
        )
        print("✅ Action audit note posted to Intercom successfully")
        
    except Exception as e:
        print(f"⚠️  Failed to post action note to Intercom: {e}")
        # Don't fail the action execution if note posting fails

