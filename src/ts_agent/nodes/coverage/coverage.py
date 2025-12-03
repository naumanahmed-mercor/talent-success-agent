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
from src.utils.debug import dump_prompt_to_file, dump_response_to_file
from pydantic import ValidationError as PydanticValidationError
from jsonschema import validate, ValidationError as JSONSchemaValidationError

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
    Analyze data coverage using LLM with retry logic for malformed responses.
    
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
    max_pydantic_retries = 2
    pydantic_error_text = None
    max_param_retries = 1
    param_error_text = None
    
    for pydantic_attempt in range(max_pydantic_retries + 1):
        # Log retry attempts
        if pydantic_attempt > 0:
            logger.info(f"Coverage retry attempt {pydantic_attempt + 1}/{max_pydantic_retries + 1}")
        
        # Format available data summary
        available_data_summary = _summarize_accumulated_data_with_content(
            tool_data, 
            docs_data, 
            plan_reasoning,
            planned_action_tools,
            actions_taken,
            max_actions,
            executed_actions,
            hop_number,
            max_hops,
            conversation_id
        )
        
        # Format procedure if available
        procedure_text = format_procedure_for_prompt(selected_procedure)
        
        # Get prompt from LangSmith
        prompt_template_text = get_prompt(PROMPT_NAMES["COVERAGE_NODE"])
        
        # Build context with any error feedback
        context_parts = [available_data_summary]
        if pydantic_error_text:
            context_parts.append("\n" + pydantic_error_text)
        if param_error_text:
            context_parts.append("\n" + param_error_text)
        
        full_context = "\n".join(context_parts)
        
        # Format the prompt with variables
        prompt = prompt_template_text.format(
            conversation_history=conversation_history,
            user_details=user_details,
            procedure=procedure_text,
            available_data=full_context
        )
        
        # Debug: Dump full prompt to file if DEBUG_PROMPTS env var is set
        metadata = {
            "Prompt Length": f"{len(prompt)} characters",
            "Pydantic Attempt": pydantic_attempt + 1,
            "Has Procedure": bool(selected_procedure),
            "Hop": f"{hop_number}/{max_hops}",
            "Actions Taken": f"{actions_taken}/{max_actions}"
        }
        if pydantic_error_text:
            metadata["Has Pydantic Error"] = True
        if param_error_text:
            metadata["Has Param Error"] = True
        
        suffix = ""
        if pydantic_attempt > 0:
            suffix = f"_retry{pydantic_attempt}"
        
        dump_prompt_to_file(prompt, "coverage", metadata=metadata, suffix=suffix)
        
        # Get LLM response with structured output
        llm = planner_llm()
        llm_with_structure = llm.with_structured_output(CoverageResponse, method="function_calling")
        
        try:
            coverage_response = llm_with_structure.invoke(prompt)
            
            # Debug: Dump LLM response to file if DEBUG_PROMPTS env var is set
            import time
            response_data = {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "pydantic_attempt": pydantic_attempt + 1,
                "data_sufficient": coverage_response.data_sufficient,
                "reasoning": coverage_response.reasoning,
                "confidence": coverage_response.confidence,
                "next_action": coverage_response.next_action,
                "missing_data": [
                    {
                        "gap_type": gap.gap_type,
                        "description": gap.description
                    }
                    for gap in coverage_response.missing_data
                ],
                "action_decision": None,
                "metadata": {
                    "hop": f"{hop_number}/{max_hops}",
                    "actions_taken": f"{actions_taken}/{max_actions}",
                    "prompt_length": len(prompt),
                    "has_procedure": bool(selected_procedure)
                }
            }
            
            if coverage_response.action_decision:
                response_data["action_decision"] = {
                    "action_tool_name": coverage_response.action_decision.action_tool_name,
                    "reasoning": coverage_response.action_decision.reasoning,
                    "parameters": coverage_response.action_decision.parameters
                }
            
            response_suffix = "_response"
            if pydantic_attempt > 0:
                response_suffix += f"_retry{pydantic_attempt}"
            
            dump_response_to_file(response_data, "coverage", suffix=response_suffix)
            
            # If we're executing an action, sanitize and validate parameters
            if coverage_response.next_action == "execute_action" and coverage_response.action_decision:
                action_decision = coverage_response.action_decision
                action_tool_name = action_decision.action_tool_name
                
                # Find the tool schema from planned action tools
                action_tool = next(
                    (tool for tool in (planned_action_tools or []) if tool.get("tool_name") == action_tool_name),
                    None
                )
                
                if action_tool and action_tool.get("tool_schema"):
                    input_schema = action_tool["tool_schema"].get("inputSchema", {})
                    
                    # FIRST: Sanitize parameters (inject runtime values like conversation_id)
                    from src.utils.sanitization import sanitize_tool_params
                    
                    injection_map = {
                        "conversation_id": str(conversation_id) if conversation_id else "",
                    }
                    
                    try:
                        # Sanitize (inject) parameters BEFORE validation
                        sanitized_params = sanitize_tool_params(
                            action_decision.parameters,
                            input_schema,
                            action_tool_name,
                            injection_map
                        )
                        
                        # Update the action decision with sanitized params
                        coverage_response.action_decision.parameters = sanitized_params
                        
                        # THEN: Validate the sanitized parameters against schema
                        if input_schema and input_schema.get("properties"):
                            validate(instance=sanitized_params, schema=input_schema)
                        
                        # Parameters are valid - success!
                        if pydantic_attempt > 0:
                            logger.info(f"✅ Coverage response valid after {pydantic_attempt} retry attempts")
                            print(f"   ✅ Coverage generated valid response after {pydantic_attempt} retry(s)")
                        
                        # Check if we've hit max hops - override to escalate if needed
                        if not coverage_response.data_sufficient and hop_number >= max_hops:
                            print(f"⚠️  Maximum hops ({max_hops}) reached - escalating instead of gathering more")
                            coverage_response.next_action = "escalate"
                            coverage_response.escalation_reason = f"Exceeded maximum hops ({max_hops}). Unable to gather sufficient data."
                        
                        return coverage_response
                        
                    except Exception as e:
                        # Sanitization or validation failed
                        error_details = _format_param_validation_error(e, action_tool_name, input_schema) if isinstance(e, JSONSchemaValidationError) else str(e)
                        logger.warning(f"⚠️  Action parameters invalid (Pydantic attempt {pydantic_attempt + 1}): {error_details}")
                        print(f"   ⚠️  Action parameters validation failed: {error_details}")
                        
                        # If this was the last Pydantic attempt, re-raise
                        if pydantic_attempt >= max_pydantic_retries:
                            logger.error(f"❌ Action parameters still invalid after {max_pydantic_retries} retries")
                            print(f"   ❌ Unable to generate valid action parameters after {max_pydantic_retries + 1} attempts")
                            raise ValueError(f"Action parameter validation failed: {error_details}")
                        
                        # Prepare parameter error feedback for retry
                        param_error_text = _format_param_error_for_prompt(e, action_tool_name, input_schema) if isinstance(e, JSONSchemaValidationError) else f"Parameter error: {str(e)}"
                        print(f"   🔄 Retrying Coverage call with parameter error feedback...")
                        # Continue to next iteration
                        continue
            
            # No action or no validation needed - success!
            if pydantic_attempt > 0:
                logger.info(f"✅ Coverage response valid after {pydantic_attempt} retry attempts")
                print(f"   ✅ Coverage generated valid response after {pydantic_attempt} retry(s)")
            
            # Check if we've hit max hops - override to escalate if needed
            if not coverage_response.data_sufficient and hop_number >= max_hops:
                print(f"⚠️  Maximum hops ({max_hops}) reached - escalating instead of gathering more")
                coverage_response.next_action = "escalate"
                coverage_response.escalation_reason = f"Exceeded maximum hops ({max_hops}). Unable to gather sufficient data."
            
            return coverage_response
            
        except PydanticValidationError as e:
            # LLM returned malformed data (missing fields, wrong types, etc.)
            error_details = _format_pydantic_errors(e)
            logger.warning(f"⚠️  Coverage response malformed (attempt {pydantic_attempt + 1}/{max_pydantic_retries + 1}): {error_details}")
            print(f"   ⚠️  Coverage response validation failed: {error_details}")
            
            # If this was the last attempt, re-raise the error
            if pydantic_attempt >= max_pydantic_retries:
                logger.error(f"❌ Coverage response still malformed after {max_pydantic_retries} retries")
                print(f"   ❌ Unable to generate valid Coverage response after {max_pydantic_retries + 1} attempts")
                raise
            
            # Otherwise, prepare error feedback for retry
            pydantic_error_text = _format_pydantic_error_for_prompt(e)
            print(f"   🔄 Retrying Coverage call with error feedback...")
            # Continue to next iteration




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


def _format_pydantic_errors(validation_error: PydanticValidationError) -> str:
    """
    Format Pydantic validation errors into a simple string for logging.
    
    Args:
        validation_error: Pydantic ValidationError
        
    Returns:
        Human-readable error summary
    """
    errors = validation_error.errors()
    error_messages = []
    
    for err in errors:
        location = " -> ".join(str(loc) for loc in err['loc'])
        error_type = err['type']
        message = err['msg']
        error_messages.append(f"{location}: {message} (type={error_type})")
    
    return "; ".join(error_messages)


def _format_pydantic_error_for_prompt(validation_error: PydanticValidationError) -> str:
    """
    Format Pydantic validation errors for the LLM prompt to help it fix issues.
    
    Args:
        validation_error: Pydantic ValidationError
        
    Returns:
        Formatted string with error details and instructions
    """
    errors = validation_error.errors()
    
    error_lines = [
        "=" * 80,
        "🚨 RESPONSE FORMAT ERROR",
        "=" * 80,
        "Your previous response had formatting errors. Fix the following:\n"
    ]
    
    for i, err in enumerate(errors, 1):
        location = " -> ".join(str(loc) for loc in err['loc'])
        message = err['msg']
        error_type = err['type']
        
        error_lines.append(f"ERROR {i}: {location}")
        error_lines.append(f"  Problem: {message}")
        
        if error_type == 'missing':
            error_lines.append(f"  ❌ Field '{err['loc'][-1]}' is REQUIRED but missing")
        
        error_lines.append("")
    
    error_lines.extend([
        "REQUIRED STRUCTURE:",
        "{",
        '  "data_sufficient": true/false,',
        '  "missing_data": [...],',
        '  "reasoning": "string",',
        '  "confidence": 0.0-1.0,',
        '  "next_action": "continue|gather_more|execute_action|escalate",',
        '  "escalation_reason": "string (if escalate)",',
        '  "action_decision": {',
        '    "action_tool_name": "string",',
        '    "reasoning": "string",',
        '    "parameters": {...}',
        '  }',
        "}",
        "=" * 80,
        ""
    ])
    
    return "\n".join(error_lines)


def _format_param_validation_error(
    validation_error: JSONSchemaValidationError,
    tool_name: str,
    input_schema: Dict[str, Any]
) -> str:
    """Format parameter validation error for logging."""
    return f"{tool_name}: {validation_error.message}"


def _format_param_error_for_prompt(
    validation_error: JSONSchemaValidationError,
    tool_name: str,
    input_schema: Dict[str, Any]
) -> str:
    """
    Format parameter validation errors for the LLM prompt.
    
    Args:
        validation_error: JSON Schema ValidationError
        tool_name: Name of the action tool
        input_schema: Tool's input schema
        
    Returns:
        Formatted string with error details
    """
    import json
    
    error_lines = [
        "=" * 80,
        "⚠️ ACTION PARAMETER VALIDATION ERROR",
        "=" * 80,
        f"Your parameters for '{tool_name}' failed validation.\n",
        f"ERROR: {validation_error.message}\n",
        "REQUIRED PARAMETERS:"
    ]
    
    # Show required parameters from schema
    properties = input_schema.get("properties", {})
    required_params = input_schema.get("required", [])
    
    for param in required_params:
        param_schema = properties.get(param, {})
        param_type = param_schema.get("type", "unknown")
        param_desc = param_schema.get("description", "")
        error_lines.append(f"  • {param} ({param_type}): {param_desc}")
    
    error_lines.extend([
        "\nPlease provide ALL required parameters with correct values based on gathered data.",
        "=" * 80,
        ""
    ])
    
    return "\n".join(error_lines)
