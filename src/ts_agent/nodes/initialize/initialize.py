"""
Initialize node for setting up conversation data and available tools.
"""

import os
import time
from typing import Dict, Any
from ts_agent.types import State, ToolType
from src.clients.intercom import IntercomClient
from src.mcp.factory import create_mcp_client
from .schemas import InitializeData


def initialize_node(state: State) -> State:
    """
    Initialize the state by fetching conversation data from Intercom and MCP tools.
    
    Data source behavior:
    1. Check if BOTH messages AND user_details are provided
       - If yes: Use both directly (skip Intercom fetch)
    2. Else: Fetch from Intercom
       - If messages not provided: Use messages from Intercom
       - If user_details not provided: Use user_details from Intercom
       - Always use subject from Intercom
    
    Args:
        state: Current state with conversation_id and optional data overrides
        
    Returns:
            Updated state with conversation data and available tools
    """
    # Get conversation ID and Melvin admin ID FIRST (before any potential failures)
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        state["error"] = "conversation_id is required"
        initialize_data = InitializeData(
            conversation_id="",
            messages_count=0,
            user_name=None,
            user_email=None,
            subject=None,
            tools_count=0,
            melvin_admin_id="",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            success=False,
            error="conversation_id is required"
        )
        state["initialize"] = initialize_data.model_dump()
        return state
    
    # Set conversation_id in state immediately
    state["conversation_id"] = conversation_id
    
    # Get Melvin admin ID and set it in state immediately
    melvin_admin_id = os.getenv("MELVIN_ADMIN_ID")
    if melvin_admin_id:
        state["melvin_admin_id"] = melvin_admin_id
    
    # Check for test mode (automatically sets dry_run)
    mode = state.get("mode")
    is_test_mode = mode == "test"
    
    # If test mode, set dry_run flag (unless explicitly overridden)
    if is_test_mode and "dry_run" not in state:
        state["dry_run"] = True
        print(f"🧪 Test mode enabled - dry_run set to True")
    
    # Log procedure_id if provided (already in state from inputs)
    procedure_id = state.get("procedure_id")
    if procedure_id:
        print(f"📋 Procedure ID provided: {procedure_id}")
    
    # Only initialize if not already done
    if "available_tools" not in state or state["available_tools"] is None:
        try:
            
            print(f"📞 Fetching conversation data from Intercom: {conversation_id}")
            
            # Initialize Intercom client
            intercom_api_key = os.getenv("INTERCOM_API_KEY")
            if not intercom_api_key:
                raise ValueError("INTERCOM_API_KEY environment variable is required")
            
            # Pass dry_run to IntercomClient if in test mode
            dry_run = state.get("dry_run", False)
            intercom_client = IntercomClient(intercom_api_key, dry_run=dry_run)
            
            # Check if both messages and user_details are provided
            has_messages = "messages" in state and state["messages"]
            has_user_details = "user_details" in state and state["user_details"]
            
            # If both are provided, use them directly (skip Intercom fetch)
            if has_messages and has_user_details:
                print(f"✅ Using provided messages and user details from state")
                state["subject"] = state.get("subject", "")
            else:
                # Fetch data from Intercom
                print(f"📞 Fetching conversation data from Intercom: {conversation_id}")
                conversation_data = intercom_client.get_conversation_data_for_agent(conversation_id)
                
                # Validate: Either messages or subject must be present
                has_intercom_messages = conversation_data.get("messages") and len(conversation_data["messages"]) > 0
                has_subject = conversation_data.get("subject") and conversation_data["subject"].strip()
                
                if not has_intercom_messages and not has_subject:
                    raise ValueError(f"No messages or subject found in conversation {conversation_id}")
                
                # Use Intercom data if not provided in state
                if not has_messages:
                    state["messages"] = conversation_data.get("messages", [])
                    print(f"✅ Using {len(state['messages'])} message(s) from Intercom")
                else:
                    print(f"✅ Using {len(state['messages'])} provided message(s)")
                
                if not has_user_details:
                    state["user_details"] = {
                        "name": conversation_data.get("user_name"),
                        "email": conversation_data.get("user_email")
                    }
                    print(f"✅ Using user details from Intercom")
                else:
                    print(f"✅ Using provided user details from state")
                
                # Always get subject from Intercom if we fetched
                state["subject"] = conversation_data.get("subject") or ""
            
            # Log final values
            user_details = state.get("user_details", {})
            print(f"✅ User name: {user_details.get('name') or 'None'}")
            print(f"✅ User email: {user_details.get('email') or 'None'}")
            print(f"✅ Subject: {state.get('subject', 'None')}")
            print(f"✅ Melvin admin ID: {melvin_admin_id}")
            
            # Initialize MCP client with mode-based auth token
            print("🔌 Initializing MCP client...")
            mcp_client = create_mcp_client(mode=mode)
            
            # Fetch available tools from MCP server
            print("🔧 Fetching available tools from MCP server...")
            available_tools = mcp_client.list_tools()
            
            # Filter out search_procedures tool (handled by procedure node)
            available_tools = [tool for tool in available_tools if tool.get("name") != "search_procedures"]
            
            print(f"✅ Found {len(available_tools)} available tools")
            
            # Assign tool types to each tool
            # Currently, the MCP server doesn't return tool types yet, so we assign them manually
            for tool in available_tools:
                tool_name = tool.get("name", "")
                if tool_name == "match_and_link_conversation_to_ticket":
                    tool["tool_type"] = ToolType.INTERNAL_ACTION.value
                else:
                    tool["tool_type"] = ToolType.GATHER.value
            
            print(f"🏷️  Assigned tool types:")
            for tool in available_tools:
                print(f"   {tool.get('name')}: {tool.get('tool_type')}")
            
            # Initialize state with proper values (NO MCP client in state)
            state["available_tools"] = available_tools
            state["tool_data"] = state.get("tool_data", {})
            state["docs_data"] = state.get("docs_data", {})
            state["hops"] = state.get("hops", [])
            state["max_hops"] = state.get("max_hops", 3)
            state["actions"] = state.get("actions", [])
            state["max_actions"] = state.get("max_actions", 1)
            state["actions_taken"] = state.get("actions_taken", 0)
            state["response"] = state.get("response", "")
            state["error"] = state.get("error", None)
            state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Store initialize data using Pydantic model
            initialize_data = InitializeData(
                conversation_id=conversation_id,
                messages_count=len(state["messages"]),
                user_name=state.get("user_details", {}).get("name"),
                user_email=state.get("user_details", {}).get("email"),
                subject=state.get("subject"),
                tools_count=len(available_tools),
                melvin_admin_id=melvin_admin_id,
                timestamp=state["timestamp"],
                success=True,
                error=None
            )
            state["initialize"] = initialize_data.model_dump()
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            error_msg = f"Initialization failed: {str(e)}"
            state["error"] = error_msg
            state["escalation_reason"] = error_msg
            state["response"] = "Sorry, I'm unable to connect to the required services right now."
            
            # Store error in initialize data
            initialize_data = InitializeData(
                conversation_id=state.get("conversation_id", ""),
                messages_count=0,
                user_name=None,
                user_email=None,
                subject=None,
                tools_count=0,
                melvin_admin_id="",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                success=False,
                error=error_msg
            )
            state["initialize"] = initialize_data.model_dump()
    
    return state

