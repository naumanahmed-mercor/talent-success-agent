#!/usr/bin/env python3
"""
Script to retrieve the execution status and full state of a LangSmith thread.

Usage:
    python scripts/get_thread_state.py <thread_id>
    python scripts/get_thread_state.py 807b7a88-3ddc-4396-b29b-ff1ddf931662 --save
    python scripts/get_thread_state.py 807b7a88-3ddc-4396-b29b-ff1ddf931662 --verbose
    
    # You can also pass a full LangSmith URL:
    python scripts/get_thread_state.py "https://smith.langchain.com/o/.../t/807b7a88-3ddc-4396-b29b-ff1ddf931662"
"""

import os
import sys
import json
import argparse
import re
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from langsmith import Client
except ImportError:
    print("❌ langsmith package not found. Please install it with: pip install langsmith")
    sys.exit(1)


def extract_ids_from_url(url: str) -> Optional[Dict[str, str]]:
    """
    Extract thread ID, project ID, and organization ID from a LangSmith URL.
    
    Args:
        url: LangSmith URL (e.g., https://smith.langchain.com/o/{org_id}/projects/p/{project_id}/t/{thread_id})
        
    Returns:
        Dictionary with extracted IDs, or None if URL is not recognized
    """
    # Pattern for LangSmith URLs
    # Example: https://smith.langchain.com/o/91eda327-8ae7-460d-87dc-6ba6d6054560/projects/p/555adc72-45db-401c-b0df-d0626422a8f1/t/807b7a88-3ddc-4396-b29b-ff1ddf931662
    pattern = r'smith\.langchain\.com/o/([^/]+)/projects/p/([^/]+)/t/([^/]+)'
    
    match = re.search(pattern, url)
    if match:
        return {
            "organization_id": match.group(1),
            "project_id": match.group(2),
            "thread_id": match.group(3),
        }
    
    # Try alternative pattern without projects
    # Example: https://smith.langchain.com/o/{org_id}/t/{thread_id}
    pattern2 = r'smith\.langchain\.com/o/([^/]+)/t/([^/]+)'
    match2 = re.search(pattern2, url)
    if match2:
        return {
            "organization_id": match2.group(1),
            "thread_id": match2.group(2),
        }
    
    return None


def format_timestamp(ts: Any) -> str:
    """Format timestamp for display."""
    if not ts:
        return "N/A"
    
    try:
        if isinstance(ts, str):
            # Try to parse ISO format
            from dateutil import parser as date_parser
            dt = date_parser.parse(ts)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass
    
    return str(ts)


def print_run_info(run: Any, verbose: bool = False):
    """Print detailed information about a run."""
    print(f"\n{'='*80}")
    print("🏃 RUN INFORMATION")
    print("="*80)
    
    # Basic info - handle both dict and object access
    run_id = getattr(run, "id", run.get("id") if isinstance(run, dict) else None)
    run_name = getattr(run, "name", run.get("name", "N/A") if isinstance(run, dict) else "N/A")
    run_type = getattr(run, "run_type", run.get("run_type", "N/A") if isinstance(run, dict) else "N/A")
    status = getattr(run, "status", run.get("status", "N/A") if isinstance(run, dict) else "N/A")
    error = getattr(run, "error", run.get("error") if isinstance(run, dict) else None)
    
    print(f"📋 Run ID: {run_id}")
    print(f"📝 Name: {run_name}")
    print(f"🔧 Type: {run_type}")
    
    # Status with emoji
    status_emoji = {
        "success": "✅",
        "error": "❌",
        "pending": "⏳",
        "running": "🏃",
    }
    print(f"{status_emoji.get(status, '❓')} Status: {status}")
    
    if error:
        print(f"❌ Error: {error}")
    
    # Timing info
    start_time = getattr(run, "start_time", run.get("start_time") if isinstance(run, dict) else None)
    end_time = getattr(run, "end_time", run.get("end_time") if isinstance(run, dict) else None)
    
    if start_time:
        print(f"⏰ Start Time: {format_timestamp(start_time)}")
    if end_time:
        print(f"⏱️  End Time: {format_timestamp(end_time)}")
    
    # Execution time
    if start_time and end_time:
        try:
            from dateutil import parser as date_parser
            start_dt = date_parser.parse(start_time) if isinstance(start_time, str) else start_time
            end_dt = date_parser.parse(end_time) if isinstance(end_time, str) else end_time
            duration = (end_dt - start_dt).total_seconds()
            print(f"⏲️  Duration: {duration:.2f}s")
        except:
            pass
    
    # Input/Output info
    inputs = getattr(run, "inputs", run.get("inputs", {}) if isinstance(run, dict) else {})
    outputs = getattr(run, "outputs", run.get("outputs", {}) if isinstance(run, dict) else {})
    
    if inputs:
        print(f"\n📥 Inputs:")
        for key, value in inputs.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"   • {key}: {value}")
            else:
                print(f"   • {key}: {type(value).__name__}")
    
    if outputs and verbose:
        print(f"\n📤 Outputs:")
        for key, value in outputs.items():
            if isinstance(value, (str, int, float, bool)):
                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"   • {key}: {display_value}")
            else:
                print(f"   • {key}: {type(value).__name__}")
    
    # Token usage
    total_tokens = getattr(run, "total_tokens", run.get("total_tokens") if isinstance(run, dict) else None)
    prompt_tokens = getattr(run, "prompt_tokens", run.get("prompt_tokens") if isinstance(run, dict) else None)
    completion_tokens = getattr(run, "completion_tokens", run.get("completion_tokens") if isinstance(run, dict) else None)
    
    if total_tokens or prompt_tokens or completion_tokens:
        print(f"\n💰 Token Usage:")
        if prompt_tokens:
            print(f"   • Prompt Tokens: {prompt_tokens:,}")
        if completion_tokens:
            print(f"   • Completion Tokens: {completion_tokens:,}")
        if total_tokens:
            print(f"   • Total Tokens: {total_tokens:,}")
    
    # Tags and metadata
    tags = getattr(run, "tags", run.get("tags", []) if isinstance(run, dict) else [])
    if tags:
        print(f"\n🏷️  Tags: {', '.join(tags)}")
    
    if verbose:
        extra = getattr(run, "extra", run.get("extra", {}) if isinstance(run, dict) else {})
        metadata = extra.get("metadata", {}) if isinstance(extra, dict) else {}
        if metadata:
            print(f"\n📊 Metadata:")
            for key, value in metadata.items():
                print(f"   • {key}: {value}")


def get_run_via_rest_api(run_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Get run directly via the LangSmith REST API.
    
    Args:
        run_id: Run UUID
        api_key: LangSmith API key
        
    Returns:
        Run data or None
    """
    try:
        url = f"https://api.smith.langchain.com/api/v1/runs/{run_id}"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        print(f"🔍 Trying REST API endpoint...")
        print(f"   URL: {url}")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Got run from REST API")
            return response.json()
        else:
            print(f"❌ REST API returned {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ Error calling REST API: {e}")
        return None


def get_thread_preview_api(thread_id: str, session_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Get thread preview using the LangSmith REST API directly.
    
    Args:
        thread_id: Thread UUID
        session_id: Session UUID (project ID)
        api_key: LangSmith API key
        
    Returns:
        Thread preview data or None
    """
    try:
        url = f"https://api.smith.langchain.com/threads/{thread_id}/preview"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        params = {
            "session_id": session_id
        }
        
        print(f"🔍 Trying thread preview API endpoint...")
        print(f"   URL: {url}")
        print(f"   Session ID: {session_id}")
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            print(f"✅ Got thread preview from API")
            return response.json()
        else:
            print(f"❌ Thread preview API returned {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ Error calling thread preview API: {e}")
        return None


def get_thread_state(thread_id: str, project_id: Optional[str] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Retrieve the execution status and state for a thread.
    
    Args:
        thread_id: The LangSmith thread ID (can be a run ID or trace ID)
        project_id: Optional project ID to filter runs
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing thread information and state, or None if error
    """
    try:
        # Initialize LangSmith client
        api_key = os.getenv("LANGSMITH_API_KEY")
        client = Client(api_key=api_key)
        
        print(f"\n🔍 Fetching thread information for: {thread_id}")
        if project_id:
            print(f"📁 Project ID: {project_id}")
        
        # Strategy 1: Try REST API directly (most reliable)
        rest_api_data = get_run_via_rest_api(thread_id, api_key)
        if rest_api_data:
            print(f"\n{'='*80}")
            print("📦 RUN DATA FROM REST API")
            print("="*80)
            
            # Parse the response
            run_data = rest_api_data
            
            # Extract key information
            run_id = run_data.get('id')
            name = run_data.get('name', 'N/A')
            status = run_data.get('status', 'N/A')
            error = run_data.get('error')
            start_time = run_data.get('start_time')
            end_time = run_data.get('end_time')
            
            # Status emoji
            status_emoji = {
                "success": "✅",
                "error": "❌",
                "pending": "⏳",
                "running": "🏃",
            }
            
            print(f"\n📋 Run ID: {run_id}")
            print(f"📝 Name: {name}")
            print(f"{status_emoji.get(status, '❓')} Status: {status}")
            
            if error:
                print(f"❌ Error: {error}")
            
            if start_time:
                print(f"⏰ Start Time: {start_time}")
            if end_time:
                print(f"⏱️  End Time: {end_time}")
            
            # Get outputs
            outputs = run_data.get('outputs', {})
            inputs = run_data.get('inputs', {})
            
            if outputs:
                print(f"\n{'='*80}")
                print("📦 THREAD STATE")
                print("="*80)
                
                # Common state fields
                conversation_id = outputs.get("conversation_id")
                response = outputs.get("response")
                error_msg = outputs.get("error")
                hops = outputs.get("hops", [])
                selected_procedure = outputs.get("selected_procedure")
                escalation_reason = outputs.get("escalation_reason")
                finalize_data = outputs.get("finalize", {})
                
                if conversation_id:
                    print(f"💬 Conversation ID: {conversation_id}")
                
                if selected_procedure:
                    procedure_title = selected_procedure.get("title") or selected_procedure.get("id")
                    print(f"📚 Procedure: {procedure_title}")
                
                print(f"🔄 Hops: {len(hops)}")
                
                if finalize_data:
                    melvin_status = finalize_data.get("melvin_status", "unknown")
                    print(f"📊 Melvin Status: {melvin_status}")
                
                if escalation_reason:
                    print(f"🚨 Escalation: {escalation_reason}")
                
                if error_msg:
                    print(f"❌ Error: {error_msg}")
                
                if response and not verbose:
                    print(f"\n📝 Response Preview:")
                    print("-" * 80)
                    preview_length = 300
                    if len(response) > preview_length:
                        print(response[:preview_length] + "...")
                    else:
                        print(response)
                elif response and verbose:
                    print(f"\n📝 Full Response:")
                    print("-" * 80)
                    print(response)
                
                # Token usage
                total_tokens = run_data.get("total_tokens")
                prompt_tokens = run_data.get("prompt_tokens")
                completion_tokens = run_data.get("completion_tokens")
                
                if total_tokens or prompt_tokens or completion_tokens:
                    print(f"\n💰 Token Usage:")
                    if prompt_tokens:
                        print(f"   • Prompt Tokens: {prompt_tokens:,}")
                    if completion_tokens:
                        print(f"   • Completion Tokens: {completion_tokens:,}")
                    if total_tokens:
                        print(f"   • Total Tokens: {total_tokens:,}")
                
                if verbose:
                    print(f"\n📤 Full Run Data:")
                    print(json.dumps(run_data, indent=2, default=str))
                
                # Return structured data
                return {
                    "run_id": run_id,
                    "trace_id": run_data.get('trace_id'),
                    "project_id": project_id,
                    "name": name,
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": error,
                    "inputs": inputs,
                    "outputs": outputs,
                    "state": outputs,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tags": run_data.get('tags', []),
                    "metadata": run_data.get('extra', {}).get('metadata', {}),
                    "source": "rest_api"
                }
            else:
                print(f"\n⚠️  No outputs found in run")
                return {
                    "run_id": run_id,
                    "name": name,
                    "status": status,
                    "error": error,
                    "inputs": inputs,
                    "source": "rest_api"
                }
        
        # Strategy 2: Try SDK read_run method
        try:
            print(f"\n🔍 Trying SDK read_run method...")
            run = client.read_run(thread_id)
            
            if run:
                print(f"✅ Found run directly")
                # Print run information
                print_run_info(run, verbose=verbose)
                
                # Get the full state from outputs
                outputs = run.outputs or {}
                
                if outputs:
                    print(f"\n{'='*80}")
                    print("📦 THREAD STATE")
                    print("="*80)
                    
                    # Try to extract state information
                    state = outputs
                    
                    # Common state fields
                    conversation_id = state.get("conversation_id")
                    response = state.get("response")
                    error = state.get("error")
                    hops = state.get("hops", [])
                    selected_procedure = state.get("selected_procedure")
                    escalation_reason = state.get("escalation_reason")
                    finalize_data = state.get("finalize", {})
                    
                    if conversation_id:
                        print(f"💬 Conversation ID: {conversation_id}")
                    
                    if selected_procedure:
                        procedure_title = selected_procedure.get("title") or selected_procedure.get("id")
                        print(f"📚 Procedure: {procedure_title}")
                    
                    print(f"🔄 Hops: {len(hops)}")
                    
                    if finalize_data:
                        melvin_status = finalize_data.get("melvin_status", "unknown")
                        print(f"📊 Melvin Status: {melvin_status}")
                    
                    if escalation_reason:
                        print(f"🚨 Escalation: {escalation_reason}")
                    
                    if error:
                        print(f"❌ Error: {error}")
                    
                    if response and not verbose:
                        print(f"\n📝 Response Preview:")
                        print("-" * 80)
                        preview_length = 300
                        if len(response) > preview_length:
                            print(response[:preview_length] + "...")
                        else:
                            print(response)
                    elif response and verbose:
                        print(f"\n📝 Full Response:")
                        print("-" * 80)
                        print(response)
                    
                    # Return full data
                    return {
                        "run_id": run.id,
                        "trace_id": run.trace_id or thread_id,
                        "project_id": project_id,
                        "name": run.name,
                        "status": run.status,
                        "start_time": run.start_time,
                        "end_time": run.end_time,
                        "error": run.error,
                        "inputs": run.inputs or {},
                        "outputs": outputs,
                        "state": state,
                        "total_tokens": run.total_tokens,
                        "prompt_tokens": run.prompt_tokens,
                        "completion_tokens": run.completion_tokens,
                        "tags": run.tags or [],
                        "metadata": getattr(run, "extra", {}).get("metadata", {}),
                    }
                else:
                    print(f"\n⚠️  No state found in run outputs")
                    return {
                        "run_id": run.id,
                        "trace_id": run.trace_id or thread_id,
                        "project_id": project_id,
                        "name": run.name,
                        "status": run.status,
                        "error": run.error,
                        "inputs": run.inputs or {},
                    }
        except Exception as read_error:
            print(f"❌ Could not read run via SDK: {read_error}")
        
        # Strategy 3: Try the thread preview API if we have a project_id
        if project_id:
            preview_data = get_thread_preview_api(thread_id, project_id, api_key)
            if preview_data:
                print(f"\n{'='*80}")
                print("📦 THREAD PREVIEW DATA")
                print("="*80)
                print(json.dumps(preview_data, indent=2, default=str))
                
                # Try to extract useful information from preview
                previews = preview_data.get('previews', {})
                if previews and verbose:
                    print(f"\n📝 Preview Details:")
                    for key, value in previews.items():
                        print(f"   • {key}: {value}")
                
                return {
                    "thread_id": thread_id,
                    "project_id": project_id,
                    "preview_data": preview_data,
                    "source": "thread_preview_api"
                }
        
        # Strategy 4: Try to get runs by trace ID using the id parameter
        try:
            print(f"\n🔍 Looking up runs for trace ID: {thread_id}")
            
            # In LangSmith, we need to query runs with the trace ID
            # The trace ID is stored as 'id' in the run metadata
            if project_id:
                print(f"   Querying project: {project_id}")
                runs = list(client.list_runs(
                    project_id=project_id,
                    filter=f'eq(id, "{thread_id}")',
                    limit=100
                ))
                
                # If no runs found with id filter, try trace_id filter
                if not runs:
                    print(f"   No runs found with id filter, trying trace_id filter...")
                    runs = list(client.list_runs(
                        project_id=project_id,
                        filter=f'eq(trace_id, "{thread_id}")',
                        limit=100
                    ))
            else:
                # Try without project filter
                runs = list(client.list_runs(
                    filter=f'eq(id, "{thread_id}")',
                    limit=100
                ))
                
                if not runs:
                    runs = list(client.list_runs(
                        filter=f'eq(trace_id, "{thread_id}")',
                        limit=100
                    ))
            
            if runs:
                print(f"✅ Found {len(runs)} run(s) in trace")
                
                # Find the root run (the one without a parent)
                root_run = None
                for run in runs:
                    if not run.parent_run_id:
                        root_run = run
                        break
                
                # If no root run found, use the first run
                if not root_run:
                    root_run = runs[0]
                
                # Print run information
                print_run_info(root_run, verbose=verbose)
                
                # Get the full state from outputs
                outputs = root_run.outputs or {}
                
                # Check if this is a LangGraph run with state
                if outputs:
                    print(f"\n{'='*80}")
                    print("📦 THREAD STATE")
                    print("="*80)
                    
                    # Try to extract state information
                    state = outputs
                    
                    # Common state fields
                    conversation_id = state.get("conversation_id")
                    response = state.get("response")
                    error = state.get("error")
                    hops = state.get("hops", [])
                    selected_procedure = state.get("selected_procedure")
                    escalation_reason = state.get("escalation_reason")
                    finalize_data = state.get("finalize", {})
                    
                    if conversation_id:
                        print(f"💬 Conversation ID: {conversation_id}")
                    
                    if selected_procedure:
                        procedure_title = selected_procedure.get("title") or selected_procedure.get("id")
                        print(f"📚 Procedure: {procedure_title}")
                    
                    print(f"🔄 Hops: {len(hops)}")
                    
                    if finalize_data:
                        melvin_status = finalize_data.get("melvin_status", "unknown")
                        print(f"📊 Melvin Status: {melvin_status}")
                    
                    if escalation_reason:
                        print(f"🚨 Escalation: {escalation_reason}")
                    
                    if error:
                        print(f"❌ Error: {error}")
                    
                    if response and not verbose:
                        print(f"\n📝 Response Preview:")
                        print("-" * 80)
                        preview_length = 300
                        if len(response) > preview_length:
                            print(response[:preview_length] + "...")
                        else:
                            print(response)
                    elif response and verbose:
                        print(f"\n📝 Full Response:")
                        print("-" * 80)
                        print(response)
                    
                    # Count child runs by type
                    run_types = {}
                    for run in runs:
                        run_type = run.run_type
                        run_types[run_type] = run_types.get(run_type, 0) + 1
                    
                    if run_types and verbose:
                        print(f"\n🔧 Run Breakdown:")
                        for run_type, count in sorted(run_types.items()):
                            print(f"   • {run_type}: {count}")
                    
                    # Return full data
                    return {
                        "run_id": root_run.id,
                        "trace_id": thread_id,
                        "project_id": project_id,
                        "name": root_run.name,
                        "status": root_run.status,
                        "start_time": root_run.start_time,
                        "end_time": root_run.end_time,
                        "error": root_run.error,
                        "inputs": root_run.inputs or {},
                        "outputs": outputs,
                        "state": state,
                        "total_tokens": root_run.total_tokens,
                        "prompt_tokens": root_run.prompt_tokens,
                        "completion_tokens": root_run.completion_tokens,
                        "tags": root_run.tags or [],
                        "metadata": getattr(root_run, "extra", {}).get("metadata", {}),
                        "child_runs_count": len(runs) - 1,
                        "run_types": run_types,
                    }
                else:
                    print(f"\n⚠️  No state found in run outputs")
                    return {
                        "run_id": root_run.id,
                        "trace_id": thread_id,
                        "project_id": project_id,
                        "name": root_run.name,
                        "status": root_run.status,
                        "error": root_run.error,
                        "inputs": root_run.inputs or {},
                    }
            else:
                # No runs found by trace ID
                print(f"❌ No trace found with ID: {thread_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error querying trace: {e}")
            return None
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def save_state_to_file(state: Dict[str, Any], thread_id: str, output_dir: Path) -> Path:
    """Save the state to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"thread_{thread_id}_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, default=str)
    
    return filepath


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Retrieve execution status and state from a LangSmith thread",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get thread status and state
  python scripts/get_thread_state.py 807b7a88-3ddc-4396-b29b-ff1ddf931662

  # Get thread state with verbose output
  python scripts/get_thread_state.py 807b7a88-3ddc-4396-b29b-ff1ddf931662 --verbose

  # Get thread state and save to file
  python scripts/get_thread_state.py 807b7a88-3ddc-4396-b29b-ff1ddf931662 --save

  # Use full LangSmith URL (will extract thread ID and project ID)
  python scripts/get_thread_state.py "https://smith.langchain.com/o/.../projects/p/.../t/807b7a88-3ddc-4396-b29b-ff1ddf931662"
  
  # Save to custom directory
  python scripts/get_thread_state.py 807b7a88-3ddc-4396-b29b-ff1ddf931662 --save --output-dir ./my_states
        """
    )
    
    parser.add_argument(
        "thread_id_or_url",
        help="LangSmith thread ID (run/trace ID) or full LangSmith URL to retrieve"
    )
    
    parser.add_argument(
        "-p", "--project-id",
        help="Project ID to filter runs (optional, will be extracted from URL if provided)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output including full response and metadata"
    )
    
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save the state to a JSON file"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("local_runs"),
        help="Directory to save state files [default: local_runs]"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ LANGSMITH_API_KEY environment variable is required")
        print("   Please set it in your .env file or export it")
        sys.exit(1)
    
    # Extract thread ID and project ID
    thread_id = args.thread_id_or_url
    project_id = args.project_id
    
    # Check if input is a URL
    if "smith.langchain.com" in args.thread_id_or_url:
        print(f"🔗 Parsing LangSmith URL...")
        ids = extract_ids_from_url(args.thread_id_or_url)
        if ids:
            thread_id = ids.get("thread_id")
            if not project_id and "project_id" in ids:
                project_id = ids.get("project_id")
            print(f"✅ Extracted thread ID: {thread_id}")
            if project_id:
                print(f"✅ Extracted project ID: {project_id}")
        else:
            print("⚠️  Could not parse URL, using as-is")
    
    # Get thread state
    state_data = get_thread_state(thread_id, project_id=project_id, verbose=args.verbose)
    
    if not state_data:
        sys.exit(1)
    
    # Save to file if requested
    if args.save:
        args.output_dir.mkdir(exist_ok=True)
        output_file = save_state_to_file(state_data, thread_id, args.output_dir)
        print(f"\n💾 State saved to: {output_file}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

