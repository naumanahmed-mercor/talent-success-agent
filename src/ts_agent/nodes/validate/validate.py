"""
Validate node for response validation.

This node sends the draft response to a validation endpoint and:
1. Adds validation results as a note to the Intercom conversation
2. Loops back to draft once if validation fails (giving draft a chance to fix issues)
3. Escalates if validation fails on the second attempt
4. Routes to response node if validation passes
"""

import os
import json
import requests
from typing import Dict, Any
from .schemas import ValidationResponse, ValidateData
from src.clients.intercom import IntercomClient


def validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the draft response against policy and intent classification.

    Args:
        state: Current state containing the draft response

    Returns:
        Updated state with validation results
    """
    print("🔍 Validate Node: Validating draft response...")
    
    # Get max retries from state (default: 1)
    max_validation_retries = state.get("max_validation_retries", 1)
    
    # Get existing validation array if available (for retry tracking)
    existing_validations = state.get("validate", [])
    if not isinstance(existing_validations, list):
        existing_validations = []
    
    # Determine current retry count based on number of previous validations
    current_retry_count = len(existing_validations)
    
    print(f"📊 Validation attempt {current_retry_count + 1}/{max_validation_retries + 1}")
    
    # Initialize validate data using Pydantic model (will be appended to array)
    validate_data = ValidateData(
        validation_response=None,
        overall_passed=False,
        validation_note_added=False,
        escalation_reason=None,
        next_action="escalate",
        retry_count=current_retry_count
    )

    try:
        # Get the draft response
        response_text = state.get("response", "")
        
        if not response_text:
            raise ValueError("No response text found to validate")

        # Get MCP configuration from environment
        mcp_base_url = os.getenv("MCP_BASE_URL")
        if not mcp_base_url:
            raise ValueError("MCP_BASE_URL environment variable is required")
        
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        if not mcp_auth_token:
            raise ValueError("MCP_AUTH_TOKEN environment variable is required")
        
        # Construct validation endpoint URL
        validation_endpoint = f"{mcp_base_url.rstrip('/')}/talent-success/melvin-validation/validate"

        print(f"📤 Sending response to validation endpoint: {validation_endpoint}")
        
        # Send validation request with Bearer token authentication
        validation_payload = {"reply": response_text}
        
        response = requests.post(
            validation_endpoint,
            json=validation_payload,
            timeout=180,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {mcp_auth_token}"
            }
        )
        
        response.raise_for_status()
        validation_result = response.json()
        
        print(f"✅ Validation response received in {validation_result.get('processing_time_ms', 0):.2f}ms")
        
        # Store raw validation results first
        validate_data.validation_response = validation_result
        
        # Add raw validation results as a note to Intercom conversation
        conversation_id = state.get("conversation_id")
        admin_id = state.get("melvin_admin_id")
        
        if conversation_id and admin_id:
            try:
                intercom_api_key = os.getenv("INTERCOM_API_KEY")
                if not intercom_api_key:
                    raise ValueError("INTERCOM_API_KEY environment variable is required")
                
                # Initialize Intercom client with dry_run from state
                dry_run = state.get("dry_run", False)
                intercom_client = IntercomClient(intercom_api_key, dry_run=dry_run)
                
                # Always use raw JSON format for the note
                overall_status = "✅ PASSED" if validation_result.get("overall_passed") else "❌ FAILED"
                note_text = f"🔍 Response Validation Results\n\n**Status**: {overall_status}\n\n```json\n{json.dumps(validation_result, indent=2)}\n```"
                
                # Add note to conversation
                intercom_client.add_note(
                    conversation_id=conversation_id,
                    note_body=note_text,
                    admin_id=admin_id
                )
                
                validate_data.validation_note_added = True
                print("✅ Validation results added as note to Intercom conversation")
                
            except Exception as e:
                print(f"⚠️  Failed to add validation note to Intercom: {e}")
                # Continue even if note fails
        
        # Parse validation response for routing logic (only care about overall_passed)
        validation_response = ValidationResponse(**validation_result)
        validate_data.overall_passed = validation_response.overall_passed
        
        # Determine next action based on validation result
        if validation_response.overall_passed:
            validate_data.next_action = "response"
            state["next_node"] = "response"
            print("✅ Validation passed - routing to response node")
        else:
            # Check if we can still retry
            if current_retry_count < max_validation_retries:
                # Retry available - loop back to draft with validation feedback
                validate_data.next_action = "draft"
                state["next_node"] = "draft"
                
                print(f"⚠️  Validation failed (attempt {current_retry_count + 1}/{max_validation_retries + 1}) - looping back to draft with feedback")
            else:
                # No more retries - escalate
                validate_data.next_action = "escalate"
                state["next_node"] = "escalate"
                
                # Simple escalation reason
                escalation_reason = f"Validation failed after {max_validation_retries + 1} attempts - see validation notes for details"
                validate_data.escalation_reason = escalation_reason
                state["escalation_reason"] = escalation_reason
                
                print(f"❌ Validation failed (attempt {current_retry_count + 1}/{max_validation_retries + 1}) - escalating")

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        print(f"❌ {error_msg}")
        validate_data.escalation_reason = error_msg
        validate_data.next_action = "escalate"
        state["error"] = error_msg
        state["next_node"] = "escalate"
        state["escalation_reason"] = error_msg

    # Append validate data to array (convert to dict for state)
    if "validate" not in state or not isinstance(state["validate"], list):
        state["validate"] = []
    state["validate"].append(validate_data.model_dump())
    
    print(f"🎯 Validate node completed - next action: {validate_data.next_action}")
    print(f"📊 Total validation attempts: {len(state['validate'])}")
    
    return state
