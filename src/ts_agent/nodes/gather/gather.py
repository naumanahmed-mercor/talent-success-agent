"""Gather node implementation for executing tool calls."""

import json
import time
from typing import Dict, Any, List
from ts_agent.types import State
from .schemas import GatherData, ToolCall, ToolResult, GatherRequest
from src.mcp.factory import create_mcp_client


def gather_node(state: State) -> State:
    """
    Execute all planned tool calls and gather results.

    This node takes the tool calls from the plan node and executes them
    using the MCP client, storing the results in the state.
    """
    # Get current hop data
    hops_array = state.get("hops", [])
    current_hop_index = len(hops_array) - 1
    
    if current_hop_index < 0:
        state["error"] = "No current hop data found"
        return state
    
    current_hop_data = hops_array[current_hop_index]
    plan_data = current_hop_data.get("plan", {})
    
    # Get only gather tool calls (plan node already separated them)
    gather_tool_calls = plan_data.get("gather_tool_calls", [])
    user_email = state.get("user_email")
    
    if not gather_tool_calls:
        # No tools needed - this is normal for simple queries like "Hi"
        print("ℹ️  No gather tools needed for this query")
        
        # Store empty gather results using GatherData TypedDict
        gather_data: GatherData = {
            "tool_results": [],
            "total_execution_time_ms": 0.0,
            "success_rate": 1.0,  # 100% success since no tools failed
            "execution_status": "completed"
        }
        current_hop_data["gather"] = gather_data
        
        print("✅ No tool execution needed - proceeding to coverage analysis")
        return state
    
    try:
        # Create MCP client (don't store in state due to serialization issues)
        # Get mode from state for auth token selection
        mode = state.get("mode")
        mcp_client = create_mcp_client(mode=mode)
        
        # Execute all tool calls
        results = []
        successful_tools = []
        failed_tools = []
        total_start_time = time.time()
        
        print(f"🔧 Executing {len(gather_tool_calls)} gather tool calls...")
        
        for i, tool_call_data in enumerate(gather_tool_calls, 1):
            start_time = time.time()  # Move start_time outside try block
            tool_call = None  # Initialize tool_call variable
            try:
                # Create ToolCall object for validation
                tool_call = ToolCall(**tool_call_data)
                
                print(f"   {i}. Executing {tool_call.tool_name}...")
                
                # Execute the tool
                result_data = _execute_tool(mcp_client, tool_call)
                
                # Special case: Add instructions for get_user_referrals
                if tool_call.tool_name == "get_user_referrals":
                    result_data = _add_referral_instructions(result_data)
                
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Create successful result
                result = ToolResult(
                    tool_name=tool_call.tool_name,
                    success=True,
                    data=result_data,
                    execution_time_ms=execution_time,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                )
                
                results.append(result)
                successful_tools.append(tool_call.tool_name)
                print(f"      ✅ Success ({execution_time:.1f}ms)")
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # Create failed result
                result = ToolResult(
                    tool_name=tool_call.tool_name if tool_call else "unknown",
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                )
                
                results.append(result)
                failed_tools.append(tool_call.tool_name if tool_call else "unknown")
                print(f"      ❌ Failed: {e}")
        
        total_execution_time = (time.time() - total_start_time) * 1000
        success_rate = len(successful_tools) / len(gather_tool_calls) if gather_tool_calls else 0
        
        # Store results in nested gather structure using GatherData TypedDict
        gather_data: GatherData = {
            "tool_results": [result.model_dump() for result in results],
            "total_execution_time_ms": total_execution_time,
            "success_rate": success_rate,
            "execution_status": "completed"
        }
        current_hop_data["gather"] = gather_data
        
        # Store individual tool results at state level (independent of hops)
        # Initialize data storage if it doesn't exist
        if "tool_data" not in state:
            state["tool_data"] = {}
        if "docs_data" not in state:
            state["docs_data"] = {}
        
        for result in results:
            if result.success and result.data:
                # Check if this is a docs tool (search_talent_docs)
                if result.tool_name == "search_talent_docs":
                    # Store docs data separately with unique key (query + hop)
                    # Extract query from the result data structure (now parsed from JSON-RPC)
                    query = "unknown_query"
                    if result.data:
                        # After parsing, result.data is a dict with 'query' field
                        if isinstance(result.data, dict):
                            query = result.data.get("query", "unknown_query")
                        # Fallback for old list format (shouldn't happen with parsing)
                        elif isinstance(result.data, list) and len(result.data) > 0:
                            first_item = result.data[0]
                            if isinstance(first_item, dict):
                                query = first_item.get("query", "unknown_query")
                    
                    # Create unique key with hop number to avoid overwriting
                    current_hop = len(state.get("hops", []))
                    unique_key = f"{query} (hop {current_hop})"
                    state["docs_data"][unique_key] = result.data
                else:
                    # Store regular tool data - accumulate if tool called multiple times
                    if result.tool_name in state["tool_data"]:
                        # Tool already called - accumulate results in a list
                        existing = state["tool_data"][result.tool_name]
                        if not isinstance(existing, list):
                            existing = [existing]
                        existing.append(result.data)
                        state["tool_data"][result.tool_name] = existing
                    else:
                        # First call to this tool
                        state["tool_data"][result.tool_name] = result.data
        
        print(f"🎉 Tool execution complete!")
        print(f"   ✅ Successful: {len(successful_tools)}")
        print(f"   ❌ Failed: {len(failed_tools)}")
        print(f"   ⏱️  Total time: {total_execution_time:.1f}ms")
        print(f"   📊 Success rate: {success_rate:.1%}")
        
    except Exception as e:
        error_msg = f"Gather node error: {str(e)}"
        state["error"] = error_msg
        state["escalation_reason"] = error_msg
        state["next_node"] = "escalate"
        print(f"❌ Gather node error: {e}")
    
    return state


def _add_referral_instructions(result_data: Any) -> Any:
    """
    Add instructions to get_user_referrals response to guide coverage/plan nodes.
    
    Args:
        result_data: Parsed result from get_user_referrals tool
        
    Returns:
        Result data augmented with instructions field
    """
    instructions = (
        "To fetch detailed application information for a specific referral, use the "
        "'get_referee_applications' tool with the 'referral_id' parameter from the referrals list above. "
        "This will return all job applications made by that referee, including referral bonus amounts "
        "locked at application time and application statuses."
    )
    
    # Since data is now parsed, it should be a dict
    if isinstance(result_data, dict):
        result_data["instructions"] = instructions
    elif isinstance(result_data, list) and len(result_data) > 0:
        # Handle list format (less common after parsing)
        for item in result_data:
            if isinstance(item, dict):
                item["instructions"] = instructions
                break  # Add to first dict only
    
    return result_data


def _parse_mcp_result(raw_result: Any) -> Any:
    """
    Parse MCP JSON-RPC formatted result to extract actual data.
    
    MCP returns data in format: {"type": "text", "text": "<json_string>"}
    This function extracts and parses the text field.
    
    Args:
        raw_result: Raw result from MCP tool call
        
    Returns:
        Parsed data (original structure if not JSON-RPC format)
    """
    
    # Handle list of results (e.g., [{"type": "text", "text": "..."}])
    if isinstance(raw_result, list) and len(raw_result) > 0:
        parsed_items = []
        for item in raw_result:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                text_content = item["text"]
                # Try to parse the JSON string
                try:
                    if isinstance(text_content, str):
                        parsed_items.append(json.loads(text_content))
                    else:
                        # Already parsed
                        parsed_items.append(text_content)
                except json.JSONDecodeError:
                    # Not valid JSON, keep as string
                    parsed_items.append(text_content)
            else:
                # Not JSON-RPC format, keep as is
                parsed_items.append(item)
        
        # If single item, unwrap the list
        if len(parsed_items) == 0:
            # Empty result - return empty list rather than failing
            return []
        return parsed_items[0] if len(parsed_items) == 1 else parsed_items
    
    # Handle single result object {"type": "text", "text": "..."}
    elif isinstance(raw_result, dict) and raw_result.get("type") == "text" and "text" in raw_result:
        text_content = raw_result["text"]
        try:
            if isinstance(text_content, str):
                return json.loads(text_content)
            else:
                # Already parsed
                return text_content
        except json.JSONDecodeError:
            # Not valid JSON, keep as string
            return text_content
    
    # Not JSON-RPC format, return as is
    return raw_result


def _execute_tool(mcp_client, tool_call: ToolCall) -> Dict[str, Any]:
    """
    Execute a single tool call using MCP client.
    
    Args:
        mcp_client: MCP client instance
        tool_call: Tool call to execute
        
    Returns:
        Tool execution result data (parsed from JSON-RPC format)
    """
    tool_name = tool_call.tool_name
    parameters = tool_call.parameters
    
    # Execute tool using MCP client
    try:
        raw_result = mcp_client.call_tool(tool_name, parameters)
        # Parse JSON-RPC formatted result to extract actual data
        parsed_result = _parse_mcp_result(raw_result)
        return parsed_result
    except json.JSONDecodeError as e:
        raise Exception(f"Tool execution failed - JSON parse error: {str(e)}")
    except Exception as e:
        # Log the actual error type and message for debugging
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else repr(e)
        raise Exception(f"Tool execution failed ({error_type}): {error_msg}")
