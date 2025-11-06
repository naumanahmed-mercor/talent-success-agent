#!/usr/bin/env python3
"""
Batch runner for testing multiple conversations in parallel.

Features:
- Parallel execution with configurable workers
- Automatic state dumping
- CSV summary generation
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ts_agent.graph import build_graph
from clients.intercom import IntercomClient


def get_first_user_message(conversation_id: str, intercom_client: IntercomClient) -> dict:
    """
    Fetch only the first user message from a conversation.
    
    Args:
        conversation_id: Intercom conversation ID
        intercom_client: IntercomClient instance
        
    Returns:
        Dictionary with conversation data containing only the first user message
    """
    # Get full conversation data
    conv_data = intercom_client.get_conversation_data_for_agent(conversation_id)
    
    # Find the first user message
    messages = conv_data.get("messages", [])
    first_user_message = None
    
    for msg in messages:
        if msg.get("role") == "user":
            first_user_message = msg
            break
    
    if not first_user_message:
        print(f"⚠️  No user message found in conversation {conversation_id}")
        return conv_data
    
    # Return data with only the first user message
    return {
        "messages": [first_user_message],
        "user_email": conv_data.get("user_email"),
        "user_name": conv_data.get("user_name"),
        "subject": conv_data.get("subject"),
        "conversation_id": conversation_id
    }


def run_conversation(conversation_id: str, output_dir: Path, first_message_only: bool = True) -> dict:
    """
    Run agent on a single conversation.
    
    Args:
        conversation_id: Intercom conversation ID
        output_dir: Directory to save state files
        first_message_only: Whether to use only first message
        
    Returns:
        Dictionary with result metadata and final state
    """
    start_time = time.time()
    
    print(f"🚀 Processing: {conversation_id}")
    
    try:
        # Initialize Intercom client
        intercom_api_key = os.getenv("INTERCOM_API_KEY")
        if not intercom_api_key:
            raise ValueError("INTERCOM_API_KEY environment variable is required")
        
        intercom_client = IntercomClient(intercom_api_key)
        
        # Optionally get first message only
        if first_message_only:
            conv_data = get_first_user_message(conversation_id, intercom_client)
            if not conv_data.get("messages"):
                print(f"❌ No messages found in {conversation_id}")
                return {
                    "conversation_id": conversation_id,
                    "success": False,
                    "error": "No messages found",
                    "elapsed_time": time.time() - start_time,
                    "state": None
                }
        
        # Build and run graph
        graph = build_graph()
        initial_state = {
            "conversation_id": conversation_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        final_state = graph.invoke(initial_state)
        
        # Save state to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"state_{conversation_id}_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(final_state, f, indent=2, default=str)
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Completed: {conversation_id} ({elapsed_time:.2f}s)")
        
        return {
            "conversation_id": conversation_id,
            "success": True,
            "error": None,
            "elapsed_time": elapsed_time,
            "output_file": str(filepath),
            "state": final_state
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Failed: {conversation_id} - {e}")
        
        return {
            "conversation_id": conversation_id,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
            "state": None
        }


def extract_csv_data(result: dict) -> dict:
    """
    Extract CSV fields from result.
    
    Args:
        result: Result dictionary with state
        
    Returns:
        Dictionary with CSV fields
    """
    conversation_id = result["conversation_id"]
    state = result.get("state")
    
    if not state:
        return {
            "conversation_id": conversation_id,
            "messages": "",
            "procedure_title": "",
            "response": f"ERROR: {result.get('error', 'Unknown error')}"
        }
    
    # Extract messages
    messages = state.get("messages", [])
    messages_text = " | ".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}"
        for msg in messages
    ])
    
    # Extract procedure title
    selected_procedure = state.get("selected_procedure")
    procedure_title = ""
    if selected_procedure:
        procedure_title = selected_procedure.get("title", selected_procedure.get("id", ""))
    
    # Extract response
    response = state.get("response", "No response generated")
    
    return {
        "conversation_id": conversation_id,
        "messages": messages_text,
        "procedure_title": procedure_title,
        "response": response
    }


def main():
    """Main function to run batch processing."""
    
    # Configuration
    conversation_ids = [
        "215471558637109",
        "215471541323221",
        "215471579653598",
        "215471523828458",
        "215471587008270",
        "215471434510806"
    ]
    
    num_workers = 3
    output_dir = Path("local_runs")
    first_message_only = True
    
    # Load environment variables
    load_dotenv()
    
    # Enable dry run mode
    os.environ["DRY_RUN"] = "true"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Print configuration
    print("\n" + "="*80)
    print("🚀 BATCH DRY RUN")
    print("="*80)
    print(f"📋 Conversations: {len(conversation_ids)}")
    print(f"👥 Workers: {num_workers}")
    print(f"📨 Mode: {'First message only' if first_message_only else 'Full conversation'}")
    print(f"🔄 Dry Run: ✅ Enabled")
    print(f"📁 Output: {output_dir.absolute()}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Run conversations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_conv = {
            executor.submit(run_conversation, conv_id, output_dir, first_message_only): conv_id
            for conv_id in conversation_ids
        }
        
        # Wait for completion and collect results
        for future in as_completed(future_to_conv):
            conv_id = future_to_conv[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Exception for {conv_id}: {e}")
                results.append({
                    "conversation_id": conv_id,
                    "success": False,
                    "error": str(e),
                    "elapsed_time": 0,
                    "state": None
                })
    
    total_time = time.time() - start_time
    
    # Generate CSV summary
    csv_file = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id", "messages", "procedure_title", "response"])
        writer.writeheader()
        
        for result in sorted(results, key=lambda x: x["conversation_id"]):
            csv_data = extract_csv_data(result)
            writer.writerow(csv_data)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 BATCH SUMMARY")
    print("="*80)
    print(f"✅ Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"❌ Failed: {sum(1 for r in results if not r['success'])}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📁 States saved to: {output_dir.absolute()}")
    print(f"📄 CSV summary: {csv_file.absolute()}")
    print("="*80)
    
    # Print individual results
    print("\n📋 Individual Results:")
    for result in sorted(results, key=lambda x: x["conversation_id"]):
        status = "✅" if result["success"] else "❌"
        print(f"\n{status} {result['conversation_id']}")
        print(f"   Time: {result['elapsed_time']:.2f}s")
        if result["success"]:
            state = result.get("state", {})
            hops = len(state.get("hops", []))
            procedure = state.get("selected_procedure")
            print(f"   Hops: {hops}")
            if procedure:
                print(f"   Procedure: {procedure.get('title', procedure.get('id'))}")
        else:
            print(f"   Error: {result['error']}")
    
    print("\n" + "="*80)
    print("✨ Batch run complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

