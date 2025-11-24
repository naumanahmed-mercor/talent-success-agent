"""Coverage node implementation for analyzing data sufficiency."""

import os
import re
import logging
from typing import Dict, Any, List, Optional
from ts_agent.types import State, ToolType
from .schemas import CoverageData, CoverageResponse, DataGap
from ts_agent.llm import planner_llm
from src.clients.prompts import get_prompt, PROMPT_NAMES
from src.utils.prompts import build_conversation_and_user_context, format_procedure_for_prompt

logger = logging.getLogger(__name__)


def coverage_node(state: State) -> State:
    """
    Analyze data coverage and determine next action.
    
    This node analyzes whether we have sufficient data to answer the user's query
    and decides whether to continue, gather more data, or escalate.
    """
    # Get current hop data - use direct reference like gather node does
    if "hops" not in state or not state["hops"]:
        state["error"] = "No current hop data found"
        return state
    
    hops_array = state.get("hops", [])
    current_hop_index = len(hops_array) - 1
    current_hop_data = hops_array[current_hop_index]
    hop_number = current_hop_data.get("hop_number", current_hop_index + 1)
    max_hops = state.get("max_hops", 3)
    
    print(f"🔍 Coverage Analysis (Hop {hop_number}/{max_hops})")
    print("=" * 50)
    
    # Note: Hop limit check will be done at the end after coverage analysis
    
    # Extract data from state level (accumulated across all hops)
    tool_data = state.get("tool_data", {})  # All accumulated tool data
    docs_data = state.get("docs_data", {})  # All accumulated docs data
    
    # Show accumulated data
    if tool_data:
        print(f"📊 Accumulated Tool Data ({len(tool_data)} tools):")
        for tool_name, data in tool_data.items():
            print(f"   📋 {tool_name}: {type(data).__name__}")
    else:
        print("📊 No accumulated tool data available")
    
    if docs_data:
        print(f"📚 Accumulated Docs Data ({len(docs_data)} queries):")
        for query, data in docs_data.items():
            print(f"   📖 {query}: {type(data).__name__}")
    else:
        print("📚 No accumulated docs data available")
    
    try:
        # Build formatted conversation history and user details (with validation)
        formatted_context = build_conversation_and_user_context(state)
    except ValueError as e:
        state["error"] = str(e)
        return state
    
    # Get action tools directly from plan (plan node already separated them)
    plan_data = current_hop_data.get("plan", {})
    planned_action_tools = plan_data.get("action_tool_calls", [])
    
    # Get available tools from state to fetch full schemas
    available_tools = state.get("available_tools", [])
    
    # Get procedure-required action tools (these are auto-added even if Plan didn't include them)
    procedure_required_action_tools = state.get("procedure_required_action_tools", [])
    
    # If there are procedure-required action tools that Plan didn't include, add them
    if procedure_required_action_tools:
        print(f"\n⚡ Checking procedure-required action tools:")
        planned_tool_names = [tc.get("tool_name") for tc in planned_action_tools]
        
        for required_tool_name in procedure_required_action_tools:
            if required_tool_name not in planned_tool_names:
                print(f"   📌 Auto-adding '{required_tool_name}' (required by procedure, missing from Plan)")
                # Add the tool to planned_action_tools with empty parameters (Coverage will generate them)
                planned_action_tools.append({
                    "tool_name": required_tool_name,
                    "parameters": {},
                    "reasoning": f"Required by procedure (auto-added)"
                })
            else:
                print(f"   ✓ '{required_tool_name}' already in Plan")
    
    # Enrich planned action tools with full schemas from available_tools
    enriched_action_tools = _enrich_action_tools_with_schemas(planned_action_tools, available_tools)
    
    # Get conversation_id from state to include in context
    conversation_id = state.get("conversation_id", "")
    
    # Coverage should ONLY see action tools that Plan suggested (with proper parameters)
    # Don't show all available action tools - Plan has the validation logic
    
    # Get executed actions from state
    executed_actions = state.get("actions", [])
    actions_taken = state.get("actions_taken", 0)
    max_actions = state.get("max_actions", 1)
    
    try:
        # Get plan reasoning from current hop (if available)
        plan_reasoning = None
        if current_hop_data.get("plan"):
            plan_reasoning = current_hop_data["plan"].get("reasoning")
        
        # Perform coverage analysis with full conversation history
        # Pass enriched action tools with full schemas
        coverage_response = _analyze_coverage(
            formatted_context["conversation_history"],
            formatted_context["user_details"],
            tool_data,
            docs_data,
            hop_number,
            max_hops,
            plan_reasoning,
            enriched_action_tools,  # Pass enriched tools with full schemas
            actions_taken,
            max_actions,
            executed_actions,  # Pass executed actions
            state.get("selected_procedure"),  # Pass selected procedure
            conversation_id  # Pass conversation_id for context
        )
        
        # Print analysis results
        print(f"✅ Data Sufficient: {coverage_response.data_sufficient}")
        print(f"💭 Reasoning: {coverage_response.reasoning}")
        
        if coverage_response.missing_data:
            print(f"❌ Missing Data:")
            for gap in coverage_response.missing_data:
                print(f"   - {gap.gap_type}: {gap.description}")
        
        print(f"🎯 Next Action: {coverage_response.next_action}")
        
        # Route based on coverage analysis and apply business logic (max_hops, max_actions)
        if coverage_response.next_action == "gather_more":
            # Check if we've exceeded max hops before redirecting to plan
            if hop_number >= max_hops:
                print(f"⚠️  Maximum hops ({max_hops}) reached - escalating instead of gathering more")
                # Update the coverage response
                coverage_response.next_action = "escalate"
                coverage_response.escalation_reason = f"Exceeded maximum hops ({max_hops}). Unable to gather sufficient data."
                state["next_node"] = "escalate"
                state["escalation_reason"] = coverage_response.escalation_reason
            else:
                print(f"🔄 Redirecting to plan node for more data gathering...")
                state["next_node"] = "plan"
        elif coverage_response.next_action == "execute_action":
            # Check if we've exceeded max actions
            actions_taken = state.get("actions_taken", 0)
            max_actions = state.get("max_actions", 1)
            if actions_taken >= max_actions:
                print(f"⚠️  Maximum actions ({max_actions}) reached - cannot execute more actions")
                state["next_node"] = "respond"
            else:
                # Get the action decision from coverage
                action_decision = coverage_response.action_decision
                if not action_decision:
                    print(f"⚠️  Coverage decided to execute action but didn't specify which - routing to respond")
                    state["next_node"] = "respond"
                else:
                    # Find the action tool from Plan's suggestions
                    action_tool_name = action_decision.action_tool_name
                    action_tool_from_plan = next(
                        (tool for tool in enriched_action_tools if tool.get("tool_name") == action_tool_name),
                        None
                    )
                    
                    if not action_tool_from_plan:
                        print(f"⚠️  Coverage wants to execute '{action_tool_name}' but it wasn't in Plan's suggestions - routing to respond")
                        state["next_node"] = "respond"
                    else:
                        # Validate and sanitize action tool parameters
                        from src.utils.sanitization import sanitize_tool_params
                        
                        try:
                            # Get tool schema and type
                            tool_schema = action_tool_from_plan.get("tool_schema")
                            tool_type = action_tool_from_plan.get("tool_type", ToolType.ACTION.value)
                            
                            if not tool_schema:
                                print(f"⚠️  No tool schema found for '{action_tool_name}' - routing to respond")
                                state["next_node"] = "respond"
                            else:
                                # Validate and sanitize parameters
                                coverage_params = action_decision.parameters
                                conversation_id = state.get("conversation_id", "")
                                input_schema = tool_schema.get("inputSchema", {})
                                
                                # Build injection map with conversation_id (ensure it's a string)
                                injection_map = {
                                    "conversation_id": str(conversation_id) if conversation_id else "",
                                }
                                
                                sanitized_params = sanitize_tool_params(
                                    coverage_params,
                                    input_schema,
                                    action_tool_name,
                                    injection_map,
                                    tool_type=tool_type
                                )
                                
                                # Update action_decision with sanitized parameters
                                coverage_response.action_decision.parameters = sanitized_params
                                
                                print(f"⚡ Redirecting to action node to execute: {action_tool_name}")
                                print(f"   Coverage's reasoning: {action_decision.reasoning}")
                                print(f"   Sanitized parameters: {sanitized_params}")
                                
                                # Store which action tool to execute
                                state["next_node"] = "action"
                        
                        except Exception as e:
                            print(f"⚠️  Parameter validation failed for '{action_tool_name}': {e}")
                            print(f"   Routing to respond instead")
                            state["next_node"] = "respond"
        elif coverage_response.next_action == "continue":
            print(f"✅ Proceeding to response generation...")
            state["next_node"] = "respond"
        elif coverage_response.next_action == "escalate":
            print(f"🚨 Escalating to human team: {coverage_response.escalation_reason}")
            state["next_node"] = "escalate"
            state["escalation_reason"] = coverage_response.escalation_reason
        
        # Store coverage data using simplified CoverageData TypedDict
        # Convert Pydantic model to dict for state storage
        coverage_data: CoverageData = {
            "coverage_response": coverage_response.model_dump(),
            "next_node": state.get("next_node", "end")
        }
        current_hop_data["coverage"] = coverage_data
        
    except Exception as e:
        state["error"] = f"Coverage analysis failed: {str(e)}"
        state["next_node"] = "escalate"
        state["escalation_reason"] = f"Coverage analysis failed: {str(e)}"
        print(f"❌ Coverage analysis error: {e}")
    
    return state




def _enrich_action_tools_with_schemas(
    planned_action_tools: List[Dict[str, Any]],
    available_tools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich planned action tools with full schemas from available_tools.
    
    Args:
        planned_action_tools: Action tools suggested by Plan (tool_name, parameters, reasoning)
        available_tools: All available tools from MCP with full schemas
        
    Returns:
        Enriched action tools with full schemas added
    """
    enriched_tools = []
    
    for planned_tool in planned_action_tools:
        tool_name = planned_tool.get("tool_name")
        
        # Find the full tool schema from available_tools
        tool_schema = next(
            (tool for tool in available_tools if tool.get("name") == tool_name),
            None
        )
        
        # Create enriched tool with schema
        enriched_tool = planned_tool.copy()
        if tool_schema:
            enriched_tool["tool_schema"] = tool_schema
        
        enriched_tools.append(enriched_tool)
    
    return enriched_tools




def _analyze_coverage(
    conversation_history: str,
    user_details: str,
    tool_data: Dict[str, Any],
    docs_data: Dict[str, Any],
    hop_number: int,
    max_hops: int,
    plan_reasoning: Optional[str] = None,
    planned_action_tools: Optional[List[Dict[str, Any]]] = None,
    actions_taken: int = 0,
    max_actions: int = 1,
    executed_actions: Optional[List[Dict[str, Any]]] = None,
    selected_procedure: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None
) -> CoverageResponse:
    """
    Analyze data coverage using LLM.
    
    Args:
        conversation_history: Formatted conversation history string
        user_details: Formatted user details string
        tool_data: Accumulated tool data
        docs_data: Accumulated docs data
        hop_number: Current hop number
        max_hops: Maximum allowed hops
        plan_reasoning: Reasoning from plan node explaining why tools were/weren't chosen
        planned_action_tools: Action tools suggested by Plan, enriched with full schemas
        actions_taken: Number of actions taken so far
        max_actions: Maximum allowed actions
        executed_actions: List of actions that have already been executed
        selected_procedure: Optional selected procedure from RAG store
        conversation_id: Conversation ID for context (will be injected at runtime)
        
    Returns:
        Coverage analysis response
    """
    # Create detailed prompt for LLM with actual data content
    # Get prompt from LangSmith
    prompt_template_text = get_prompt(PROMPT_NAMES["COVERAGE_NODE"])
    
    # Format available data summary
    available_data_summary = _summarize_accumulated_data_with_content(
        tool_data, 
        docs_data, 
        plan_reasoning,
        planned_action_tools,  # Now includes full schemas
        actions_taken,
        max_actions,
        executed_actions,  # Pass executed actions
        hop_number,  # Add current hop
        max_hops,  # Add max hops
        conversation_id  # Add conversation_id
    )
    
    # Format procedure if available
    procedure_text = format_procedure_for_prompt(selected_procedure)
    
    # Format the prompt with variables
    prompt = prompt_template_text.format(
        conversation_history=conversation_history,
        user_details=user_details,
        procedure=procedure_text,
        available_data=available_data_summary
    )
    
    # Get LLM response with structured output
    llm = planner_llm()
    llm_with_structure = llm.with_structured_output(CoverageResponse, method="function_calling")
    
    try:
        coverage_response = llm_with_structure.invoke(prompt)
        
        # Check if we've hit max hops - override to escalate if needed
        if not coverage_response.data_sufficient and hop_number >= max_hops:
            print(f"⚠️  Maximum hops ({max_hops}) reached - escalating instead of gathering more")
            coverage_response.next_action = "escalate"
            coverage_response.escalation_reason = f"Exceeded maximum hops ({max_hops}). Unable to gather sufficient data."
        
        return coverage_response
        
    except Exception as e:
        raise ValueError(f"Failed to get coverage analysis: {e}")




def _summarize_accumulated_data_with_content(
    tool_data: Dict[str, Any], 
    docs_data: Dict[str, Any],
    plan_reasoning: Optional[str] = None,
    planned_action_tools: Optional[List[Dict[str, Any]]] = None,
    actions_taken: int = 0,
    max_actions: int = 1,
    executed_actions: Optional[List[Dict[str, Any]]] = None,
    hop_number: int = 1,
    max_hops: int = 3,
    conversation_id: Optional[str] = None
) -> str:
    """Summarize accumulated tool and docs data with actual content for LLM prompt."""
    summary = []
    
    # Hop progress at the top
    summary.append(f"CURRENT HOP: {hop_number}/{max_hops}")
    summary.append("")
    
    # Add conversation ID context if available
    if conversation_id:
        summary.append(f"CONVERSATION ID: {conversation_id}")
        summary.append("")
    
    # Plan reasoning section (helps explain why tools were/weren't chosen)
    if plan_reasoning:
        summary.append("PLAN REASONING:")
        summary.append(f"  {plan_reasoning}")
        summary.append("")  # blank line
    
    # Tool data section
    if tool_data:
        summary.append("TOOL DATA:")
        for tool_name, data in tool_data.items():
            summary.append(f"\n{tool_name}:")
            summary.extend(_format_data_content(data))
    else:
        summary.append("TOOL DATA: None available")
    
    # Docs data section
    if docs_data:
        summary.append("\nDOCS DATA:")
        for query, data in docs_data.items():
            summary.append(f"\nQuery: '{query}'")
            summary.extend(_format_data_content(data))
    else:
        summary.append("\nDOCS DATA: None available")
    
    # Executed actions section (IMPORTANT: prevents infinite loops)
    if executed_actions and len(executed_actions) > 0:
        summary.append("\n\n⚠️  EXECUTED ACTIONS:")
        summary.append(f"The following actions have already been executed in this conversation:")
        for i, action in enumerate(executed_actions, 1):
            tool_name = action.get("tool_name", "unknown")
            success = action.get("success", False)
            status = "✅ SUCCESS" if success else "❌ FAILED"
            hop_num = action.get("hop_number", "?")
            audit = action.get("audit_notes", "")
            summary.append(f"\n  {i}. {tool_name} ({status}) - Hop {hop_num}")
            summary.append(f"     Audit: {audit}")
        summary.append(f"\n⚠️  DO NOT execute these actions again - they have already been attempted!")
    
    # Planned action tools section (if Plan suggested any)
    if planned_action_tools and len(planned_action_tools) > 0:
        summary.append("\n\nAVAILABLE ACTION TOOLS:")
        summary.append(f"Actions taken so far: {actions_taken}/{max_actions}")
        summary.append("The following action tools are available for execution.")
        summary.append("If you decide to execute an action tool, you MUST provide complete parameters based on the tool schema and gathered data.")
        
        if actions_taken >= max_actions:
            summary.append(f"⚠️  Maximum actions reached - cannot execute more action tools")
        else:
            for tc in planned_action_tools:
                tool_name = tc.get('tool_name')
                summary.append(f"\n  Tool: {tool_name}")
                summary.append(f"  Plan's Reasoning: {tc.get('reasoning', 'N/A')}")
                
                # Show full tool schema if available
                tool_schema = tc.get('tool_schema')
                if tool_schema:
                    summary.append(f"  Tool Description: {tool_schema.get('description', 'N/A')}")
                    
                    # Show input schema
                    input_schema = tool_schema.get('inputSchema')
                    if input_schema:
                        summary.append(f"  Input Schema:")
                        properties = input_schema.get('properties', {})
                        required_params = input_schema.get('required', [])
                        
                        for param_name, param_info in properties.items():
                            is_required = "REQUIRED" if param_name in required_params else "optional"
                            param_type = param_info.get('type', 'any')
                            param_desc = param_info.get('description', 'No description')
                            summary.append(f"    - {param_name} ({param_type}, {is_required}): {param_desc}")
                else:
                    # Fallback to showing Plan's parameters if no schema
                    summary.append(f"  Suggested Parameters: {tc.get('parameters', {})}")
    
    return "\n".join(summary)


def _format_data_content(data: Any) -> List[str]:
    """Format data content for display - show complete data without truncation."""
    import json
    content = []
    
    if isinstance(data, list) and len(data) > 0:
        # Show all list items with complete content
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'text' in item:
                # Show complete text content
                text_content = item['text']
                content.append(f"  Item {i+1}: {text_content}")
            elif isinstance(item, dict):
                # Show complete dict content
                content.append(f"  Item {i+1}: {json.dumps(item, indent=2)}")
            else:
                content.append(f"  Item {i+1}: {str(item)}")
    elif isinstance(data, dict):
        # Show complete dict content
        content.append(f"  {json.dumps(data, indent=2)}")
    else:
        content.append(f"  {str(data)}")
    
    return content
