"""
Procedure node for retrieving and evaluating internal procedures from RAG store.
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from ts_agent.types import State
from ts_agent.llm import planner_llm
from src.mcp.factory import create_mcp_client
from src.clients.intercom import IntercomClient
from src.clients.prompts import get_prompt, PROMPT_NAMES
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
    1. Generate a query using LLM based on user's messages
    2. Fetch top-k results from procedure RAG endpoint
    3. Evaluate results using LLM to find matching procedure
    4. Store selected procedure in state if match found
    
    Args:
        state: Current state with messages and user details
        
    Returns:
        Updated state with procedure data
    """
    try:
        print("📚 Starting procedure retrieval...")
        
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
        rag_results = _fetch_procedures_from_mcp(query_result.query, top_k)
        print(f"✅ Retrieved {len(rag_results)} procedures")
        
        # Step 3: Evaluate results using LLM
        print("🧐 Evaluating procedures for match...")
        evaluation = _evaluate_procedures(messages, rag_results, query_result.query)
        print(f"✅ Match found: {evaluation.is_match}")
        print(f"   Reasoning: {evaluation.reasoning}")
        
        # Step 4: Store selected procedure if match found
        selected_procedure = None
        if evaluation.is_match and 0 <= evaluation.selected_procedure_index < len(rag_results):
            selected_result = rag_results[evaluation.selected_procedure_index]
            selected_procedure = SelectedProcedure(
                id=selected_result.id,
                title=selected_result.title,
                content=selected_result.content,
                reasoning=evaluation.reasoning,
                relevance_score=selected_result.relevance_score
            )
            print(f"✅ Selected procedure: {selected_procedure.title or selected_procedure.id}")
            
            # Store selected procedure at root level
            state["selected_procedure"] = selected_procedure.model_dump()
            
            # Log procedure selection to API
            _log_procedure_selection_to_api(
                state=state,
                selected_procedure=selected_procedure,
                query=query_result.query
            )
        else:
            print("ℹ️  No procedure selected")
            state["selected_procedure"] = None
        
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
        
        # Don't set error in state - this is a non-critical failure
        # The agent can continue without procedures
        print("⚠️  Continuing without procedure guidance")
    
    return state


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
        
        # Initialize Intercom client
        intercom_client = IntercomClient(intercom_api_key)
        
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


def _fetch_procedures_from_mcp(query: str, top_k: int = 5) -> List[ProcedureResult]:
    """
    Fetch procedures from MCP API using search_procedures tool.
    
    Args:
        query: Search query
        top_k: Number of results to fetch
        
    Returns:
        List of ProcedureResult objects
    """
    # Create MCP client
    mcp_client = create_mcp_client()
    
    try:
        # Call search_procedures tool via MCP client
        content = mcp_client.call_tool(
            tool_name="search_procedures",
            arguments={
                "query": query,
                "top_k": top_k
            },
            timeout=30.0
        )
        
        # MCP returns content as array of text/image objects
        # Find the text content containing results
        results_text = None
        for item in content:
            if item.get("type") == "text":
                results_text = item.get("text")
                break
        
        if not results_text:
            return []
        
        # Parse the JSON results from text
        import json
        results_data = json.loads(results_text)
        
        # Parse results into ProcedureResult objects
        results = []
        for item in results_data.get("results", []):
            # Construct content from the procedure structure
            content_parts = []
            
            # Add description
            if "description" in item:
                content_parts.append(f"Description: {item['description']}")
            
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
            
            # Convert ID to string if it's an integer
            proc_id = item.get("id")
            if isinstance(proc_id, int):
                proc_id = str(proc_id)
            
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
    finally:
        mcp_client.close()


def _evaluate_procedures(
    messages: List[Dict[str, Any]],
    procedures: List[ProcedureResult],
    query: str
) -> ProcedureEvaluation:
    """
    Evaluate procedures to find a perfect match for the current scenario.
    
    Args:
        messages: List of conversation messages
        procedures: List of retrieved procedures
        query: The search query used
        
    Returns:
        ProcedureEvaluation with match result and reasoning
    """
    # Format messages for context
    message_context = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in messages
    ])
    
    # Format procedures for evaluation
    procedures_text = ""
    for idx, proc in enumerate(procedures):
        procedures_text += f"\n--- Procedure {idx} ---\n"
        if proc.title:
            procedures_text += f"Title: {proc.title}\n"
        if proc.id:
            procedures_text += f"ID: {proc.id}\n"
        procedures_text += f"Content:\n{proc.content}\n"
    
    # Get prompt from LangSmith (or local file)
    system_prompt = get_prompt(PROMPT_NAMES["PROCEDURE_MATCHING"])
    
    user_prompt = f"""Conversation:
{message_context}

Search Query Used: {query}

Retrieved Procedures:
{procedures_text}

Evaluate these procedures and determine if any perfectly match this scenario."""
    
    llm = planner_llm().with_structured_output(ProcedureEvaluation)
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return response


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
        # Skip logging in dry run mode
        if os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes"):
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

