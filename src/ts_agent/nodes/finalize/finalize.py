"""
Finalize node for cleanup actions.

This node handles all final actions before ending the workflow:
- Updates Melvin Status custom attribute on Intercom
- Snoozes the conversation for 5 minutes
- In test mode: POSTs response to webhook_url
"""

import os
import time
import requests
from typing import Dict, Any
from .schemas import FinalizeData
from src.clients.intercom import IntercomClient, MelvinResponseStatus


def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalize the workflow by updating status and snoozing conversation.
    
    In test mode (mode == "test"):
    - POSTs the response to MCP webhook endpoint
    - Skips Intercom updates (dry_run=True)

    Args:
        state: Current state

    Returns:
        Updated state with finalize data
    """
    print("🏁 Finalize Node: Wrapping up...")

    # Check if test mode
    is_test_mode = state.get("mode") == "test"
    
    # In test mode, POST response to MCP webhook endpoint
    if is_test_mode:
        try:
            mcp_base_url = os.getenv("MCP_BASE_URL")
            mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
            procedure_id = state.get("procedure_id")
            
            if mcp_base_url and mcp_auth_token and procedure_id:
                print(f"🪝 Test mode: Posting test results to MCP webhook")
                
                webhook_url = f"{mcp_base_url}/talent-success/procedures/{procedure_id}/tests/webhook"
                
                payload = {
                    "conversation_id": state.get("conversation_id"),
                    "response": state.get("response", ""),
                    "escalation_reason": state.get("escalation_reason"),
                    "error": state.get("error")
                }
                
                webhook_response = requests.post(
                    webhook_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {mcp_auth_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=10
                )
                
                if webhook_response.status_code == 200:
                    print(f"✅ Test results posted to MCP webhook successfully")
                else:
                    print(f"⚠️  MCP webhook returned status {webhook_response.status_code}: {webhook_response.text}")
            else:
                missing = []
                if not mcp_base_url: missing.append("MCP_BASE_URL")
                if not mcp_auth_token: missing.append("MCP_AUTH_TOKEN")
                if not procedure_id: missing.append("procedure_id")
                print(f"⚠️  Missing required fields for webhook post: {', '.join(missing)}")
                
        except Exception as webhook_error:
            print(f"⚠️  Failed to post to MCP webhook: {webhook_error}")
            # Continue even if webhook fails

    # Determine Melvin Status based on workflow outcome
    melvin_status = _determine_melvin_status(state)
    
    # Initialize finalize data using Pydantic model
    finalize_data = FinalizeData(
        melvin_status=melvin_status.value,
        status_updated=False,
        conversation_snoozed=False,
        snooze_duration_seconds=300,  # 5 minutes
        error=None,
        webhook_posted=is_test_mode
    )

    try:
        # Get required data from state
        conversation_id = state.get("conversation_id")
        admin_id = state.get("melvin_admin_id")

        if not conversation_id or not admin_id:
            print("⚠️  Missing conversation_id or melvin_admin_id, skipping finalization")
            state["finalize"] = finalize_data.model_dump()
            state["next_node"] = "end"
            return state

        # Initialize Intercom client with dry_run mode
        intercom_api_key = os.getenv("INTERCOM_API_KEY")
        if not intercom_api_key:
            raise ValueError("INTERCOM_API_KEY environment variable is required")

        dry_run = state.get("dry_run", False)
        intercom_client = IntercomClient(intercom_api_key, dry_run=dry_run)
        
        if dry_run:
            print("🧪 Dry run mode: IntercomClient will skip all write operations")

        # Update Melvin Status custom attribute
        try:
            print(f"🔄 Updating Melvin Status to '{melvin_status.value}' for conversation {conversation_id}")
            result = intercom_client.update_conversation_custom_attribute(
                conversation_id=conversation_id,
                attribute_name="Melvin Status",
                attribute_value=melvin_status.value
            )
            # Only mark as updated if not a dry run
            if result and not result.get("dry_run", False):
                finalize_data.status_updated = True
                print("✅ Melvin Status updated successfully")
            else:
                print("🧪 [DRY RUN] Melvin Status update skipped")
        except Exception as status_error:
            print(f"⚠️  Failed to update Melvin Status: {status_error}")
            # Continue even if status update fails

        # Snooze conversation for 5 minutes
        try:
            snooze_until = int(time.time()) + finalize_data.snooze_duration_seconds
            print(f"💤 Snoozing conversation {conversation_id} for 5 minutes")
            result = intercom_client.snooze_conversation(
                conversation_id=conversation_id,
                snooze_until=snooze_until,
                admin_id=admin_id
            )
            # Only mark as snoozed if not a dry run
            if result and not result.get("dry_run", False):
                finalize_data.conversation_snoozed = True
                print("✅ Conversation snoozed successfully")
            else:
                print("🧪 [DRY RUN] Conversation snooze skipped")
        except Exception as snooze_error:
            print(f"⚠️  Failed to snooze conversation: {snooze_error}")
            # Continue even if snooze fails

    except Exception as e:
        error_msg = f"Finalization error: {str(e)}"
        print(f"❌ {error_msg}")
        finalize_data.error = error_msg

    # Store finalize data at state level (convert to dict for state)
    state["finalize"] = finalize_data.model_dump()
    state["next_node"] = "end"

    print(f"🎯 Finalize completed - status: {melvin_status.value}, snoozed: {finalize_data.conversation_snoozed}")

    return state


def _determine_melvin_status(state: Dict[str, Any]) -> MelvinResponseStatus:
    """
    Determine the appropriate Melvin Status based on workflow outcome.

    Args:
        state: Current state

    Returns:
        Appropriate MelvinResponseStatus enum value
    """
    # Check draft response type first (for ROUTE_TO_TEAM)
    draft_data = state.get("draft")
    if draft_data and draft_data.get("response_type") == "ROUTE_TO_TEAM":
        return MelvinResponseStatus.ROUTE_TO_TEAM
    
    # Check if we have escalate data (escalation occurred)
    escalate_data = state.get("escalate")
    if escalate_data:
        # Escalation occurred - determine status from escalation source
        escalation_source = escalate_data.get("escalation_source", "unknown")
        escalation_reason = state.get("escalation_reason", "")
        
        # Check if escalation is due to actions being taken (should be ROUTE_TO_TEAM)
        if escalation_source == "action":
            return MelvinResponseStatus.ROUTE_TO_TEAM
        
        # Check if user requested to talk to a human
        if "requested to talk to a human" in escalation_reason.lower():
            return MelvinResponseStatus.ROUTE_TO_TEAM
        elif escalation_source == "validate":
            return MelvinResponseStatus.VALIDATION_FAILED
        elif escalation_source == "draft":
            return MelvinResponseStatus.ROUTE_TO_TEAM  # Changed from RESPONSE_FAILED
        elif escalation_source == "coverage":
            return MelvinResponseStatus.ROUTE_TO_TEAM
        elif escalation_source == "initialization":
            return MelvinResponseStatus.ERROR
        else:
            return MelvinResponseStatus.ERROR
    
    # Check if response was delivered successfully
    response_delivery = state.get("response_delivery")
    if response_delivery:
        if response_delivery.get("intercom_delivered"):
            return MelvinResponseStatus.SUCCESS
        else:
            return MelvinResponseStatus.MESSAGE_FAILED
    
    # Default to error if we can't determine status
    return MelvinResponseStatus.ERROR
