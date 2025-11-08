#!/usr/bin/env python3
"""
List recent traces/runs from LangSmith to help test the get_thread_state script.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from langsmith import Client
except ImportError:
    print("❌ langsmith package not found. Please install it with: pip install langsmith")
    sys.exit(1)


def list_recent_traces(project_id: str = "555adc72-45db-401c-b0df-d0626422a8f1", limit: int = 10):
    """
    List recent traces from a LangSmith project.
    
    Args:
        project_id: LangSmith project ID
        limit: Number of recent traces to show
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("❌ LANGSMITH_API_KEY not found in environment")
        return
    
    client = Client(api_key=api_key)
    
    print(f"\n{'='*80}")
    print(f"📊 Fetching {limit} most recent traces from LangSmith")
    print(f"{'='*80}")
    print(f"Project ID: {project_id}\n")
    
    try:
        # Fetch root runs (traces) from the project
        runs = list(client.list_runs(
            project_id=project_id,
            limit=limit,
            is_root=True  # Only get root traces (full agent executions)
        ))
        
        if not runs:
            print("❌ No traces found in project")
            return
        
        print(f"✅ Found {len(runs)} traces\n")
        print(f"{'='*80}")
        print("RECENT TRACES")
        print(f"{'='*80}\n")
        
        for i, run in enumerate(runs, 1):
            # Extract basic info
            run_id = run.id
            trace_id = getattr(run, 'trace_id', run_id)
            run_name = run.name or "N/A"
            status = run.status or "N/A"
            
            # Get timestamps
            start_time = run.start_time
            if start_time:
                if isinstance(start_time, str):
                    from dateutil import parser as date_parser
                    start_time = date_parser.parse(start_time)
                start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_time_str = "N/A"
            
            # Get conversation ID from inputs
            inputs = run.inputs or {}
            conversation_id = inputs.get('conversation_id', 'N/A')
            
            # Get outputs
            outputs = run.outputs or {}
            has_response = bool(outputs.get('response'))
            has_error = bool(outputs.get('error') or run.error)
            
            # Status emoji
            status_emoji = {
                "success": "✅",
                "error": "❌",
                "pending": "⏳",
                "running": "🏃",
            }
            
            print(f"{i}. {status_emoji.get(status, '❓')} Trace ID: {trace_id}")
            print(f"   Run ID: {run_id}")
            print(f"   Name: {run_name}")
            print(f"   Status: {status}")
            print(f"   Conversation ID: {conversation_id}")
            print(f"   Start Time: {start_time_str}")
            print(f"   Has Response: {'Yes' if has_response else 'No'}")
            print(f"   Has Error: {'Yes' if has_error else 'No'}")
            
            # Generate URLs
            trace_url = f"https://smith.langchain.com/o/91eda327-8ae7-460d-87dc-6ba6d6054560/projects/p/{project_id}/t/{trace_id}"
            print(f"   Trace URL: {trace_url}")
            
            # Test command
            print(f"\n   💡 Test with this trace:")
            print(f"   python scripts/get_thread_state.py {trace_id} -p {project_id}")
            print(f"   or")
            print(f'   python scripts/get_thread_state.py "{trace_url}"\n')
            print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error fetching traces: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="List recent traces from LangSmith project"
    )
    parser.add_argument(
        '-p', '--project-id',
        default="555adc72-45db-401c-b0df-d0626422a8f1",
        help="LangSmith project ID (default: 555adc72-45db-401c-b0df-d0626422a8f1)"
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        default=10,
        help="Number of recent traces to show (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    list_recent_traces(args.project_id, args.limit)


if __name__ == "__main__":
    main()

