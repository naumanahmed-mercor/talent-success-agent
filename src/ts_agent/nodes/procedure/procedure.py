"""
Procedure node for retrieving and evaluating internal procedures from RAG store.
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from ts_agent.types import State, ToolType
from ts_agent.llm import planner_llm
from src.mcp.factory import create_mcp_client
from src.clients.intercom import IntercomClient
from src.clients.prompts import get_prompt, PROMPT_NAMES
from src.utils.debug import dump_prompt_to_file
from .schemas import (
    ProcedureData,
    ProcedureResult,
    SelectedProcedure,
    QueryGeneration,
    ProcedureEvaluation
)


def procedure_node(state: State) -> State:
    """
    Retrieve and evaluate procedures from RAG store.
    
    Steps:
    1. Check if procedure_id is provided in state:
       - If explicitly set to empty string or None: skip procedure retrieval entirely
       - If has valid value: fetch procedure by ID
       - If not in state: generate query and search
    2. If searching: Generate a query using LLM based on user's messages
    3. Fetch top-k results from procedure RAG endpoint (or single by ID)
    4. Evaluate results using LLM to find matching procedure
    5. Store selected procedure in state if match found
    
    Args:
        state: Current state with messages and user details
        
    Returns:
        Updated state with procedure data
    """
    try:
        print("📚 Starting procedure retrieval...")
        
        # Check if procedure_id is explicitly provided in state
        if "procedure_id" in state:
            procedure_id = state.get("procedure_id")
            
            # If procedure_id is explicitly empty or None, skip procedure retrieval
            if procedure_id == "" or procedure_id is None:
                print(f"⏭️  procedure_id explicitly set to empty/null - skipping procedure retrieval")
                state["selected_procedure"] = None
                
                # Store minimal procedure data indicating skip
                procedure_data = ProcedureData(
                    query="",
                    query_reasoning="Procedure retrieval explicitly skipped (procedure_id set to empty/null)",
                    top_k_results=[],
                    selected_procedure=None,
                    evaluation_reasoning="Procedure retrieval skipped",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    success=True,
                    error=None
                )
                state["procedure_node"] = procedure_data.model_dump()
                return state
        else:
            procedure_id = None
        
        if procedure_id:
            # Test mode: Fetch procedure by ID directly
            print(f"📋 Fetching procedure by ID: {procedure_id}")
            
            selected_result = _fetch_procedure_by_id(procedure_id)
            
            if selected_result:
                selected_procedure = SelectedProcedure(
                    id=selected_result.id,
                    title=selected_result.title,
                    content=selected_result.content,
                    reasoning=f"Procedure selected by ID: {procedure_id}",
                    relevance_score=selected_result.relevance_score
                )
                print(f"✅ Found procedure by ID: {selected_procedure.title or selected_procedure.id}")
                
                # Store selected procedure at root level
                state["selected_procedure"] = selected_procedure.model_dump()
                
                # Filter available_tools based on procedure-specific tool requirements
                _filter_procedure_specific_tools(state, selected_procedure)
                
                # Log procedure selection to API (unless in test/dry_run mode)
                _log_procedure_selection_to_api(
                    state=state,
                    selected_procedure=selected_procedure,
                    query=f"Direct fetch by ID: {procedure_id}"
                )
                
                # Store procedure data
                procedure_data = ProcedureData(
                    query=f"Direct fetch by ID: {procedure_id}",
                    query_reasoning="Procedure ID provided directly",
                    top_k_results=[selected_result],
                    selected_procedure=selected_procedure,
                    evaluation_reasoning=f"Procedure selected by ID: {procedure_id}",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    success=True,
                    error=None
                )
                state["procedure_node"] = procedure_data.model_dump()
                
                # Set custom attributes on Intercom conversation
                _set_procedure_custom_attributes(
                    state=state,
                    selected_procedure=selected_procedure
                )
                
                return state
            else:
                print(f"❌ Procedure with ID {procedure_id} not found - continuing with normal search")
                # Continue with normal search flow
                
        # Normal mode: Search for procedures
        # Get user messages
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state")
        
        # Step 1: Generate query using LLM
        print("🔍 Generating procedure search query...")
        query_result = _generate_query(messages)
        print(f"✅ Generated query: {query_result.query}")
        print(f"   Reasoning: {query_result.reasoning}")
        
        # Step 2: Fetch top-k results from MCP API
        print("📥 Fetching procedures from MCP server...")
        top_k = int(os.getenv("PROCEDURE_TOP_K", "5"))
        mode = state.get("mode")
        rag_results = _fetch_procedures_from_mcp(query_result.query, mode, top_k)
        print(f"✅ Retrieved {len(rag_results)} procedures")
        
        # Step 3: Evaluate results using LLM
        print("🧐 Evaluating procedures for match...")
        evaluation = _evaluate_procedures(messages, rag_results, query_result.query)
        print(f"✅ Match found: {evaluation.is_match}")
        print(f"   Reasoning: {evaluation.reasoning}")
        
        # Step 4: Store selected procedure if match found
        selected_procedure = None
        if evaluation.is_match and evaluation.selected_procedure_data:
            # Use the full procedure data from the select endpoint
            proc_data = evaluation.selected_procedure_data
            
            # Build content from the full procedure data
            content_parts = []
            
            if "description" in proc_data and proc_data["description"]:
                content_parts.append(f"Description: {proc_data['description']}")
            
            if "category" in proc_data and proc_data["category"]:
                content_parts.append(f"Category: {proc_data['category']}")
            
            if "tools_required" in proc_data and proc_data["tools_required"]:
                tools_str = ", ".join(proc_data["tools_required"])
                content_parts.append(f"\nTools Required: {tools_str}")
            
            if "steps" in proc_data and proc_data["steps"]:
                content_parts.append("\nSteps:")
                for i, step in enumerate(proc_data["steps"], 1):
                    content_parts.append(f"{i}. {step}")
            
            if "notes" in proc_data and proc_data["notes"]:
                content_parts.append("\nNotes:")
                if isinstance(proc_data["notes"], list):
                    for note in proc_data["notes"]:
                        content_parts.append(f"- {note}")
                else:
                    content_parts.append(f"\n{proc_data['notes']}")
            
            content = "\n".join(content_parts) if content_parts else ""
            
            # Get relevance score from search results if available
            relevance_score = None
            if 0 <= evaluation.selected_procedure_index < len(rag_results):
                relevance_score = rag_results[evaluation.selected_procedure_index].relevance_score
            
            # Get procedure ID
            proc_id = proc_data.get("id") or proc_data.get("procedure_id")
            if isinstance(proc_id, int):
                proc_id = str(proc_id)
            
            selected_procedure = SelectedProcedure(
                id=proc_id,
                title=proc_data.get("title"),
                content=content,
                reasoning=evaluation.reasoning,
                relevance_score=relevance_score
            )
            print(f"✅ Selected procedure: {selected_procedure.title or selected_procedure.id}")
            
            # Store selected procedure at root level
            state["selected_procedure"] = selected_procedure.model_dump()
            
            # Filter available_tools based on procedure-specific tool requirements
            # Also detect and store procedure-required action tools
            _filter_procedure_specific_tools(state, selected_procedure)
            
            # Log procedure selection to API
            _log_procedure_selection_to_api(
                state=state,
                selected_procedure=selected_procedure,
                query=query_result.query
            )
        else:
            print("ℹ️  No procedure selected")
            state["selected_procedure"] = None
            
            # Filter out procedure-specific tools when no procedure is selected
            _filter_procedure_specific_tools(state, None)
        
        # Store procedure data
        procedure_data = ProcedureData(
            query=query_result.query,
            query_reasoning=query_result.reasoning,
            top_k_results=rag_results,
            selected_procedure=selected_procedure,
            evaluation_reasoning=evaluation.reasoning,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            success=True,
            error=None
        )
        state["procedure_node"] = procedure_data.model_dump()
        
        print("✅ Procedure node completed successfully")
        
        # Add note to Intercom with procedure details
        _add_procedure_note_to_intercom(
            state=state,
            query=query_result.query,
            selected_procedure=selected_procedure,
            evaluation_reasoning=evaluation.reasoning
        )
        
        # Set custom attributes on Intercom conversation
        _set_procedure_custom_attributes(
            state=state,
            selected_procedure=selected_procedure
        )
        
    except Exception as e:
        print(f"❌ Failed to retrieve procedures: {e}")
        error_msg = f"Procedure retrieval failed: {str(e)}"
        
        # Store error in procedure data
        procedure_data = ProcedureData(
            query="",
            query_reasoning="",
            top_k_results=[],
            selected_procedure=None,
            evaluation_reasoning="",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            success=False,
            error=error_msg
        )
        state["procedure_node"] = procedure_data.model_dump()
        state["selected_procedure"] = None
        
        # Filter out procedure-specific tools when procedure retrieval fails
        _filter_procedure_specific_tools(state, None)
        
        # Set custom attributes to indicate no procedure was used
        _set_procedure_custom_attributes(state=state, selected_procedure=None)
        
        # Don't set error in state - this is a non-critical failure
        # The agent can continue without procedures
        print("⚠️  Continuing without procedure guidance")
    
    return state


def _filter_procedure_specific_tools(state: State, selected_procedure: Optional[SelectedProcedure]) -> None:
    """
    Filter available_tools in state based on procedure-specific tool requirements.
    
    Some tools are sensitive and should only be available when explicitly mentioned
    in the selected procedure content. If no procedure is selected, all procedure-specific
    tools are removed.
    
    Also detects which procedure-specific ACTION tools are required by the procedure
    and stores them in state['procedure_required_action_tools'] so they can be
    automatically available to Coverage even if Plan doesn't include them.
    
    Args:
        state: Current state with available_tools
        selected_procedure: The selected procedure with content to check (None if no procedure)
    """
    # Configuration: Tools that require procedure authorization
    # These tools will be removed from available_tools unless mentioned in procedure content
    PROCEDURE_SPECIFIC_TOOLS = {
        "route_conversation_to_project_client": {
            "reason": "Sensitive routing action - only available when procedure explicitly requires it",
            "search_terms": ["route_conversation_to_project_client"]
        },
        "generate_reset_interview_link": {
            "reason": "Sensitive action - only available when procedure explicitly requires it",
            "search_terms": ["generate_reset_interview_link"]
        }
    }
    
    available_tools = state.get("available_tools", [])
    if not available_tools:
        return
    
    # If no procedure, remove all procedure-specific tools and clear required action tools
    if not selected_procedure:
        tool_names = list(PROCEDURE_SPECIFIC_TOOLS.keys())
        filtered_tools = [
            tool for tool in available_tools 
            if tool.get("name") not in tool_names
        ]
        
        removed_count = len(available_tools) - len(filtered_tools)
        
        if removed_count > 0:
            state["available_tools"] = filtered_tools
            print(f"🔒 No procedure selected: filtered out {removed_count} procedure-specific tool(s)")
            print(f"   Tools removed: {', '.join(tool_names)}")
            print(f"   Total tools available: {len(filtered_tools)}")
        
        # Clear procedure-required action tools
        state["procedure_required_action_tools"] = []
        return
    
    # Procedure selected - check which tools are authorized
    procedure_content = selected_procedure.content.lower() if selected_procedure.content else ""
    
    # Debug: Log procedure content for troubleshooting
    print(f"\n🔍 DEBUG: Checking procedure content for tool authorization:")
    print(f"   Procedure: {selected_procedure.title or selected_procedure.id}")
    print(f"   Content length: {len(procedure_content)} chars")
    print(f"   Content preview (first 200 chars): {procedure_content[:200]}...")
    
    # Track which tools to remove
    tools_to_remove = []
    procedure_required_action_tools = []
    
    for tool_name, config in PROCEDURE_SPECIFIC_TOOLS.items():
        # Check if any search term is in the procedure content
        is_authorized = any(
            search_term.lower() in procedure_content 
            for search_term in config["search_terms"]
        )
        
        # Debug: Log search results
        print(f"\n   Checking '{tool_name}':")
        for search_term in config["search_terms"]:
            found = search_term.lower() in procedure_content
            print(f"      '{search_term}': {'✓ FOUND' if found else '✗ not found'}")
        
        if not is_authorized:
            tools_to_remove.append(tool_name)
            print(f"   🔒 Result: FILTERING OUT (not mentioned in procedure)")
            print(f"   Reason: {config['reason']}")
        else:
            print(f"   ✅ Result: AUTHORIZED (found in procedure)")
            
            # Check if this is an action tool and add to required list
            tool_schema = next((t for t in available_tools if t.get("name") == tool_name), None)
            if tool_schema and tool_schema.get("tool_type") == ToolType.ACTION.value:
                procedure_required_action_tools.append(tool_name)
                print(f"   ⚡ Added to procedure_required_action_tools (type: ACTION)")

    
    # Remove unauthorized tools from available_tools
    if tools_to_remove:
        filtered_tools = [
            tool for tool in available_tools 
            if tool.get("name") not in tools_to_remove
        ]
        
        removed_count = len(available_tools) - len(filtered_tools)
        state["available_tools"] = filtered_tools
        
        print(f"🔧 Filtered {removed_count} procedure-specific tool(s) from available_tools")
        print(f"   Total tools available: {len(filtered_tools)}")
    
    # Store procedure-required action tools in state
    state["procedure_required_action_tools"] = procedure_required_action_tools
    if procedure_required_action_tools:
        print(f"\n⚡ Procedure requires {len(procedure_required_action_tools)} action tool(s):")
        for tool_name in procedure_required_action_tools:
            print(f"   - {tool_name}")
        print(f"   These will be automatically available to Coverage even if Plan doesn't include them.")


def _generate_query(messages: List[Dict[str, Any]]) -> QueryGeneration:
    """
    Generate a search query for procedures based on user messages.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        QueryGeneration with query and reasoning
    """
    # Format messages for context
    message_context = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in messages
    ])
    
    system_prompt = """You are an expert at analyzing customer support conversations and finding relevant internal procedures.

Your task is to generate a SHORT, SIMPLE search query (5-10 words maximum) that will find procedures relevant to the customer's issue or request.

IMPORTANT: Keep the query extremely concise. Focus on the core topic only.

Good examples:
- "application status"
- "payment issue"
- "account verification"
- "interview scheduling"

Bad examples (too long):
- "candidate application status inquiry procedure India verify identity ATS lookup"
- "how to handle payment disputes and refund requests for contractors"

Generate a clear, SHORT query (2-5 words) that captures the main topic."""
    
    user_prompt = f"""Based on this conversation, generate a SHORT search query (2-5 words) to find relevant internal procedures:

{message_context}

Generate a SHORT query (2-5 words only) that captures the main topic."""
    
    llm = planner_llm().with_structured_output(QueryGeneration)
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return response


def _add_procedure_note_to_intercom(
    state: State,
    query: str,
    selected_procedure: Optional[SelectedProcedure],
    evaluation_reasoning: str
) -> None:
    """
    Add a note to Intercom documenting the procedure search and selection.
    
    Args:
        state: Current state with conversation_id and admin_id
        query: The search query used
        selected_procedure: The selected procedure (if any)
        evaluation_reasoning: Reasoning for selection/rejection
    """
    try:
        conversation_id = state.get("conversation_id")
        admin_id = state.get("melvin_admin_id")
        
        if not conversation_id or not admin_id:
            print("⚠️  Cannot post procedure note: missing conversation_id or admin_id")
            return
        
        # Get Intercom API key
        intercom_api_key = os.getenv("INTERCOM_API_KEY")
        if not intercom_api_key:
            print("⚠️  Cannot post procedure note: INTERCOM_API_KEY not found")
            return
        
        # Initialize Intercom client with dry_run from state
        dry_run = state.get("dry_run", False)
        intercom_client = IntercomClient(intercom_api_key, dry_run=dry_run)
        
        # Build note content
        note_lines = [
            "📚 **Procedure Search Results**",
            "",
            f"**Query:** `{query}`",
            ""
        ]
        
        if selected_procedure:
            note_lines.extend([
                "✅ **Procedure Selected:** Yes",
                f"**Title:** {selected_procedure.title}",
                f"**ID:** {selected_procedure.id}",
                "",
                "**Reasoning:**",
                evaluation_reasoning,
                "",
                "**Procedure Content:**",
                "```",
                selected_procedure.content,
                "```"
            ])
        else:
            note_lines.extend([
                "❌ **Procedure Selected:** No",
                "",
                "**Reasoning:**",
                evaluation_reasoning
            ])
        
        note_content = "\n".join(note_lines)
        
        # Post note to Intercom
        print(f"📝 Posting procedure note to Intercom conversation {conversation_id}")
        intercom_client.add_note(
            conversation_id=conversation_id,
            note_body=note_content,
            admin_id=admin_id
        )
        print("✅ Procedure note posted to Intercom successfully")
        
    except Exception as e:
        print(f"⚠️  Failed to post procedure note to Intercom: {e}")
        # Don't fail the procedure node if note posting fails


def _fetch_procedure_by_id(procedure_id: str) -> Optional[ProcedureResult]:
    """
    Fetch a single procedure by ID from MCP API.
    
    Args:
        procedure_id: The procedure ID to fetch
        
    Returns:
        ProcedureResult object or None if not found
    """
    try:
        # Get MCP configuration from environment
        mcp_base_url = os.getenv("MCP_BASE_URL")
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        
        if not mcp_base_url or not mcp_auth_token:
            raise ValueError("MCP_BASE_URL and MCP_AUTH_TOKEN must be set")
        
        # Make GET request to fetch procedure by ID
        url = f"{mcp_base_url}/talent-success/procedures/get"
        headers = {
            "Authorization": f"Bearer {mcp_auth_token}",
            "Content-Type": "application/json"
        }
        params = {"id": procedure_id}
        
        print(f"📡 Fetching procedure by ID from MCP API: {procedure_id}")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 404:
            print(f"❌ Procedure not found: {procedure_id}")
            return None
        
        response.raise_for_status()
        procedure_data = response.json()
        
        # Unwrap the procedure data if it's nested under 'procedure' key
        if "procedure" in procedure_data:
            procedure_data = procedure_data["procedure"]
        
        # Parse the procedure data into ProcedureResult
        content_parts = []
        
        # Add description
        if "description" in procedure_data:
            content_parts.append(f"Description: {procedure_data['description']}")
        
        # Add tools required
        if "tools_required" in procedure_data and procedure_data["tools_required"]:
            tools_str = ", ".join(procedure_data["tools_required"])
            content_parts.append(f"\nTools Required: {tools_str}")
        
        # Add steps
        if "steps" in procedure_data and procedure_data["steps"]:
            content_parts.append("\nSteps:")
            for i, step in enumerate(procedure_data["steps"], 1):
                content_parts.append(f"{i}. {step}")
        
        # Add notes if present
        if "notes" in procedure_data and procedure_data["notes"]:
            # Handle notes as either a list or string
            if isinstance(procedure_data["notes"], list):
                content_parts.append("\nNotes:")
                for note in procedure_data["notes"]:
                    content_parts.append(f"- {note}")
            else:
                content_parts.append(f"\nNotes:\n{procedure_data['notes']}")
        
        content = "\n".join(content_parts)
        
        result = ProcedureResult(
            id=procedure_data.get("id", procedure_id),
            title=procedure_data.get("title", ""),
            content=content,
            metadata=procedure_data.get("metadata", {}),
            relevance_score=1.0  # Direct fetch, so perfect relevance
        )
        
        print(f"✅ Successfully fetched procedure: {result.title or result.id}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch procedure by ID: {e}")
        return None
    except Exception as e:
        print(f"❌ Error parsing procedure data: {e}")
        return None


def _fetch_procedures_from_mcp(query: str, mode: Optional[str] = None, top_k: int = 5) -> List[ProcedureResult]:
    """
    Fetch procedures from MCP API using /talent-success/procedures/search endpoint.
    
    Args:
        query: Search query
        mode: Optional mode (not used for direct HTTP calls)
        top_k: Number of results to fetch
        
    Returns:
        List of ProcedureResult objects
    """
    try:
        # Get MCP configuration
        mcp_base_url = os.getenv("MCP_BASE_URL", "https://aws.api.mercor.com")
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        
        if not mcp_auth_token:
            raise ValueError("MCP_AUTH_TOKEN must be set")
        
        # Call search endpoint
        url = f"{mcp_base_url}/talent-success/procedures/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mcp_auth_token}"
        }
        payload = {
            "query": query,
            "top_k": top_k,
            "min_score": 0.3
        }
        
        print(f"📡 Calling MCP search endpoint: POST {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results_data = data.get("results", [])
        
        print(f"✅ Retrieved {len(results_data)} results from search endpoint")
        
        # Parse results into ProcedureResult objects
        results = []
        for item in results_data:
            # Construct content from the procedure structure
            content_parts = []
            
            # Add description
            if "description" in item:
                content_parts.append(f"Description: {item['description']}")
            
            # Add category
            if "category" in item:
                content_parts.append(f"Category: {item['category']}")
            
            # Add tools required
            if "tools_required" in item and item["tools_required"]:
                tools_str = ", ".join(item["tools_required"])
                content_parts.append(f"\nTools Required: {tools_str}")
            
            # Add steps
            if "steps" in item and item["steps"]:
                content_parts.append("\nSteps:")
                for i, step in enumerate(item["steps"], 1):
                    content_parts.append(f"{i}. {step}")
            
            # Add notes
            if "notes" in item and item["notes"]:
                content_parts.append("\nNotes:")
                for note in item["notes"]:
                    content_parts.append(f"- {note}")
            
            # Combine all parts
            content = "\n".join(content_parts) if content_parts else ""
            
            # Get ID - try both 'id' and 'procedure_id' fields
            proc_id = item.get("id") or item.get("procedure_id")
            if isinstance(proc_id, int):
                proc_id = str(proc_id)
            
            # Debug log if ID is missing
            if not proc_id:
                print(f"⚠️  Warning: Procedure missing ID - title: {item.get('title')}")
                print(f"   Available keys: {list(item.keys())}")
            
            result = ProcedureResult(
                id=proc_id,
                title=item.get("title"),
                content=content,
                relevance_score=item.get("similarity", item.get("score")),
                metadata={
                    "category": item.get("category"),
                    "document_type": item.get("document_type"),
                }
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        raise ValueError(f"Failed to fetch procedures from MCP API: {str(e)}")


def _evaluate_procedures(
    messages: List[Dict[str, Any]],
    procedures: List[ProcedureResult],
    query: str
) -> ProcedureEvaluation:
    """
    Evaluate procedures using /talent-success/procedures/select endpoint.
    
    Args:
        messages: List of conversation messages
        procedures: List of retrieved procedures
        query: The search query used (not used in new endpoint)
        
    Returns:
        ProcedureEvaluation with match result and reasoning
    """
    try:
        # Get MCP configuration
        mcp_base_url = os.getenv("MCP_BASE_URL", "https://aws.api.mercor.com")
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        
        if not mcp_auth_token:
            raise ValueError("MCP_AUTH_TOKEN must be set")
        
        # Convert messages to the format expected by the endpoint
        conversation_messages = []
        for msg in messages:
            conversation_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Extract procedure IDs
        procedure_ids = [proc.id for proc in procedures if proc.id]
        
        print(f"📋 Extracted {len(procedure_ids)} procedure IDs for evaluation:")
        for proc_id in procedure_ids:
            print(f"   - {proc_id}")
        
        if not procedure_ids:
            # No procedures to evaluate
            return ProcedureEvaluation(
                is_match=False,
                selected_procedure_index=-1,
                reasoning="No procedures provided for evaluation"
            )
        
        # Call select endpoint
        url = f"{mcp_base_url}/talent-success/procedures/select"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mcp_auth_token}"
        }
        payload = {
            "messages": conversation_messages,
            "procedure_ids": procedure_ids
        }
        
        print(f"📡 Calling MCP select endpoint: POST {url}")
        print(f"   Evaluating {len(procedure_ids)} procedures")
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"📋 Select response - is_match: {data.get('is_match')}")
        print(f"   Selected procedure data: {data.get('selected_procedure')}")
        
        # Parse response
        is_match = data.get("is_match", False)
        reasoning = data.get("reasoning", "")
        selected_procedure_data = data.get("selected_procedure")
        
        if is_match and selected_procedure_data:
            # Find the index of the selected procedure in our list
            # Try both 'id' and 'procedure_id' fields
            selected_id = selected_procedure_data.get("id") or selected_procedure_data.get("procedure_id")
            if isinstance(selected_id, int):
                selected_id = str(selected_id)
            
            print(f"📍 Looking for procedure with ID: {selected_id}")
            
            selected_index = -1
            for idx, proc in enumerate(procedures):
                print(f"   Checking procedure {idx}: {proc.id}")
                if proc.id == selected_id:
                    selected_index = idx
                    print(f"   ✅ Found match at index {idx}")
                    break
            
            if selected_index == -1:
                print(f"   ⚠️  Warning: Selected procedure ID '{selected_id}' not found in procedures list")
            
            return ProcedureEvaluation(
                is_match=True,
                selected_procedure_index=selected_index,
                reasoning=reasoning,
                selected_procedure_data=selected_procedure_data  # Include full data from select endpoint
            )
        else:
            return ProcedureEvaluation(
                is_match=False,
                selected_procedure_index=-1,
                reasoning=reasoning,
                selected_procedure_data=None
            )
        
    except Exception as e:
        print(f"❌ Failed to evaluate procedures via MCP: {e}")
        # Fallback to no match
        return ProcedureEvaluation(
            is_match=False,
            selected_procedure_index=-1,
            reasoning=f"Procedure evaluation failed: {str(e)}",
            selected_procedure_data=None
        )


def _log_procedure_selection_to_api(
    state: State,
    selected_procedure: SelectedProcedure,
    query: str
) -> None:
    """
    Log procedure selection to API endpoint.
    
    Args:
        state: Current state with conversation_id
        selected_procedure: The selected procedure
        query: The search query used
    """
    try:
        # Skip logging if in test mode or dry_run mode
        is_test_mode = state.get("mode") == "test"
        dry_run = state.get("dry_run", False)
        
        if is_test_mode or dry_run:
            print(f"🧪 Skipping procedure logging (test_mode={is_test_mode}, dry_run={dry_run})")
            return
        
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            print("⚠️  Cannot log procedure: missing conversation_id")
            return
        
        # Get MCP auth token for API authentication
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        if not mcp_auth_token:
            print("⚠️  Cannot log procedure: MCP_AUTH_TOKEN not configured")
            return
        
        # Get MCP base URL
        mcp_base_url = os.getenv("MCP_BASE_URL", "https://aws.api.mercor.com")
        
        # Make API request to log procedure selection
        url = f"{mcp_base_url}/talent-success/procedures/logs"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mcp_auth_token}"
        }
        payload = {
            "procedure_id": selected_procedure.id,
            "conversation_id": conversation_id
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        
        print(f"📊 Logged procedure selection to API: {selected_procedure.id}")
            
    except Exception as e:
        print(f"⚠️  Failed to log procedure to API: {e}")
        # Don't fail the procedure node if logging fails


def _set_procedure_custom_attributes(
    state: State,
    selected_procedure: Optional[SelectedProcedure]
) -> None:
    """
    Set custom attributes on the Intercom conversation for procedure tracking.
    
    Sets two custom attributes:
    - "Procedure Used": boolean indicating if a procedure was used
    - "Procedure Name": text with procedure title (empty string if not used)
    
    Args:
        state: Current state with conversation_id
        selected_procedure: The selected procedure (if any)
    """
    try:
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            print("⚠️  Cannot set procedure attributes: missing conversation_id")
            return
        
        # Get Intercom API key
        intercom_api_key = os.getenv("INTERCOM_API_KEY")
        if not intercom_api_key:
            print("⚠️  Cannot set procedure attributes: INTERCOM_API_KEY not found")
            return
        
        # Initialize Intercom client with dry_run from state
        dry_run = state.get("dry_run", False)
        intercom_client = IntercomClient(intercom_api_key, dry_run=dry_run)
        
        # Determine attribute values
        procedure_used = selected_procedure is not None
        procedure_name = selected_procedure.title if selected_procedure else ""
        
        # Set "Procedure Used" attribute
        print(f"📝 Setting 'Procedure Used' = {procedure_used}")
        intercom_client.update_conversation_custom_attribute(
            conversation_id=conversation_id,
            attribute_name="Procedure Used",
            attribute_value=procedure_used
        )
        
        # Set "Procedure Name" attribute
        print(f"📝 Setting 'Procedure Name' = '{procedure_name}'")
        intercom_client.update_conversation_custom_attribute(
            conversation_id=conversation_id,
            attribute_name="Procedure Name",
            attribute_value=procedure_name
        )
        
        print("✅ Procedure custom attributes set successfully")
        
    except Exception as e:
        print(f"⚠️  Failed to set procedure custom attributes: {e}")
        # Don't fail the procedure node if attribute setting fails

