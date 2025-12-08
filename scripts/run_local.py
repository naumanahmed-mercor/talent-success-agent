#!/usr/bin/env python3
"""
Local testing script for the Talent Success Agent.

Features:
- Dry run mode by default (no writes to Intercom)
- First message only mode by default (for faster testing)
- Automatic state dumping to local_runs/ folder
- Clean output formatting
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
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


def save_state(state: dict, conversation_id: str, output_dir: Path) -> Path:
    """
    Save the final state to a JSON file.
    
    Args:
        state: The final state to save
        conversation_id: Conversation ID
        output_dir: Directory to save the state file
        
    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"state_{conversation_id}_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, default=str)
    
    return filepath


def print_summary(state: dict, elapsed_time: float, output_file: Path):
    """Print a clean summary of the run."""
    print(f"\n{'='*80}")
    print("📊 RUN SUMMARY")
    print("="*80)
    
    # Extract key metrics
    response = state.get("response", "No response generated")
    error = state.get("error")
    escalation_reason = state.get("escalation_reason")
    hops = len(state.get("hops", []))
    selected_procedure = state.get("selected_procedure")
    finalize_data = state.get("finalize", {})
    melvin_status = finalize_data.get("melvin_status", "unknown")
    
    print(f"⏱️  Execution Time: {elapsed_time:.2f}s")
    print(f"🔄 Hops: {hops}")
    print(f"📊 Melvin Status: {melvin_status}")
    
    if selected_procedure:
        print(f"📚 Procedure: {selected_procedure.get('title') or selected_procedure.get('id')}")
    
    if error:
        print(f"❌ Error: {error}")
    
    if escalation_reason:
        print(f"🚨 Escalation: {escalation_reason}")
    
    print(f"\n📝 Response Preview:")
    print("-" * 80)
    preview_length = 300
    if len(response) > preview_length:
        print(response[:preview_length] + "...")
    else:
        print(response)
    
    print("="*80)
    print(f"💾 Full state saved to: {output_file}")
    print("="*80 + "\n")


def main():
    """Main function to run the agent locally."""
    parser = argparse.ArgumentParser(
        description="Run the Talent Success Agent locally for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (dry run, first message only)
  python scripts/run_local.py 215471618006513

  # Run with specific procedure
  python scripts/run_local.py 215471618006513 --procedure-id proc-1763728852027

  # Run with full conversation
  python scripts/run_local.py 215471618006513 --full-conversation

  # Run with actual Intercom writes (careful!)
  python scripts/run_local.py 215471618006513 --no-dry-run

  # Combine flags
  python scripts/run_local.py 215471618006513 --procedure-id proc-123 --full-conversation --no-dry-run
        """
    )
    
    parser.add_argument(
        "conversation_id",
        help="Intercom conversation ID to process"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Enable dry run mode (no writes to Intercom) [default: True]"
    )
    
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Disable dry run mode (WILL write to Intercom)"
    )
    
    parser.add_argument(
        "--first-message-only",
        action="store_true",
        default=True,
        help="Use only the first user message [default: True]"
    )
    
    parser.add_argument(
        "--full-conversation",
        action="store_false",
        dest="first_message_only",
        help="Use the full conversation history"
    )
    
    parser.add_argument(
        "--procedure-id",
        help="Procedure ID to use directly (bypasses procedure search)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_runs"),
        help="Directory to save state files [default: local_runs]"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set dry run mode
    os.environ["DRY_RUN"] = "true" if args.dry_run else "false"
    
    # Enable debug prompt dumping
    os.environ["DEBUG_PROMPTS"] = "true"
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Print configuration
    print("\n" + "="*80)
    print("🚀 TALENT SUCCESS AGENT - LOCAL TEST")
    print("="*80)
    print(f"📋 Conversation ID: {args.conversation_id}")
    print(f"🔄 Dry Run: {'✅ Enabled' if args.dry_run else '⚠️  DISABLED (will write to Intercom!)'}")
    print(f"📨 Mode: {'First message only' if args.first_message_only else 'Full conversation'}")
    if args.procedure_id:
        print(f"📚 Procedure ID: {args.procedure_id}")
    print(f"📁 Output Directory: {args.output_dir.absolute()}")
    print("="*80 + "\n")
    
    if not args.dry_run:
        print("⚠️  WARNING: Dry run is DISABLED! This will make actual changes to Intercom.")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return
        print()
    
    start_time = time.time()
    
    try:
        # Initialize Intercom client
        intercom_api_key = os.getenv("INTERCOM_API_KEY")
        if not intercom_api_key:
            raise ValueError("INTERCOM_API_KEY environment variable is required")
        
        intercom_client = IntercomClient(intercom_api_key)
        
        # Optionally override conversation data for first message only mode
        conv_data = None
        if args.first_message_only:
            print(f"📥 Fetching first user message from conversation {args.conversation_id}...")
            conv_data = get_first_user_message(args.conversation_id, intercom_client)
            
            if not conv_data.get("messages"):
                print(f"❌ No messages found in conversation {args.conversation_id}")
                return
            
            first_msg = conv_data['messages'][0]['content']
            print(f"✅ First message: {first_msg[:100]}...\n" if len(first_msg) > 100 else f"✅ First message: {first_msg}\n")
        
        # Build graph
        print("🔧 Building agent graph...")
        graph = build_graph()
        
        # Prepare initial state
        initial_state = {
            "conversation_id": args.conversation_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        # Add conversation data if using first message only
        if args.first_message_only and conv_data:
            initial_state["messages"] = conv_data["messages"]
            initial_state["user_details"] = {
                "name": conv_data.get("user_name"),
                "email": conv_data.get("user_email")
            }
            initial_state["subject"] = conv_data.get("subject")
        
        # Add procedure_id if provided
        if args.procedure_id:
            initial_state["procedure_id"] = args.procedure_id
            print(f"🔍 Using procedure ID: {args.procedure_id}\n")
        
        # Run the agent
        print(f"🤖 Running agent on conversation {args.conversation_id}...\n")
        final_state = graph.invoke(initial_state)
        
        elapsed_time = time.time() - start_time
        
        # Save state to file
        output_file = save_state(final_state, args.conversation_id, args.output_dir)
        
        # Print summary
        print_summary(final_state, elapsed_time, output_file)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        elapsed_time = time.time() - start_time
        print(f"⏱️  Ran for {elapsed_time:.2f}s before interruption\n")
        sys.exit(1)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Failed to process conversation {args.conversation_id}")
        print(f"   Error: {e}")
        print(f"   Time: {elapsed_time:.2f}s\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

