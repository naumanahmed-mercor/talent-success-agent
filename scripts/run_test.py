#!/usr/bin/env python3
"""
Test mode script for the Talent Success Agent.

Features:
- Test mode with custom messages and procedure_id
- Dry run mode by default (no writes to Intercom)
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

# Add src to path first
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Load environment variables from project root
env_path = project_root / ".env"

# Override existing env vars with .env file values
load_dotenv(dotenv_path=env_path, override=True)

from ts_agent.graph import build_graph


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
    filename = f"test_state_{conversation_id}_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, default=str)
    
    return filepath


def print_summary(state: dict, elapsed_time: float, output_file: Path):
    """Print a clean summary of the run."""
    print(f"\n{'='*80}")
    print("📊 TEST RUN SUMMARY")
    print("="*80)
    
    # Extract key metrics
    response = state.get("response", "No response generated")
    error = state.get("error")
    escalation_reason = state.get("escalation_reason")
    hops = len(state.get("hops", []))
    selected_procedure = state.get("selected_procedure")
    procedure_node = state.get("procedure_node", {})
    finalize_data = state.get("finalize", {})
    melvin_status = finalize_data.get("melvin_status", "unknown")
    
    print(f"⏱️  Execution Time: {elapsed_time:.2f}s")
    print(f"🔄 Hops: {hops}")
    print(f"📊 Melvin Status: {melvin_status}")
    
    # Procedure details
    if procedure_node:
        print(f"\n📚 Procedure Node:")
        print(f"   Query: {procedure_node.get('query', 'N/A')}")
        print(f"   Success: {procedure_node.get('success', False)}")
        
        if selected_procedure:
            print(f"\n✅ Procedure Selected:")
            print(f"   ID: {selected_procedure.get('id')}")
            print(f"   Title: {selected_procedure.get('title', 'No title')}")
            print(f"   Content Length: {len(selected_procedure.get('content', ''))} chars")
            if selected_procedure.get('reasoning'):
                print(f"   Reasoning: {selected_procedure.get('reasoning')[:100]}...")
        else:
            print(f"\n❌ No Procedure Selected")
            eval_reasoning = procedure_node.get('evaluation_reasoning', 'N/A')
            if eval_reasoning:
                print(f"   Reasoning: {eval_reasoning[:150]}...")
    
    if error:
        print(f"\n❌ Error: {error}")
    
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
    """Main function to run the agent in test mode."""
    parser = argparse.ArgumentParser(
        description="Run the Talent Success Agent in test mode with custom parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with procedure ID
  python scripts/run_test.py \\
    --conversation-id 215471712802811 \\
    --procedure-id proc-1762242929712 \\
    --message "hi mercor team, i have accidentally submitted my interview"

  # Test with multiple messages
  python scripts/run_test.py \\
    --conversation-id 215471712802811 \\
    --message "First message" \\
    --message "Second message"

  # Test without procedure ID (normal search)
  python scripts/run_test.py \\
    --conversation-id 215471712802811 \\
    --message "payment issue"
        """
    )
    
    parser.add_argument(
        "--conversation-id",
        required=True,
        help="Intercom conversation ID to process"
    )
    
    parser.add_argument(
        "--procedure-id",
        help="Procedure ID to fetch directly (test mode)"
    )
    
    parser.add_argument(
        "--message",
        action="append",
        dest="messages",
        help="Message to test with (can be specified multiple times). Use --message for each message."
    )
    
    parser.add_argument(
        "--user-email",
        help="User email for testing (optional, will use Intercom data if not provided)"
    )
    
    parser.add_argument(
        "--user-name",
        help="User name for testing (optional, will use Intercom data if not provided)"
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
        "--output-dir",
        type=Path,
        default=Path("local_runs"),
        help="Directory to save state files [default: local_runs]"
    )
    
    args = parser.parse_args()
    
    # Default message if none provided
    if not args.messages:
        args.messages = ["hi mercor team, i have accidentally submitted my interview can you rearrange it for me please?"]
    
    # Set dry run mode
    os.environ["DRY_RUN"] = "true" if args.dry_run else "false"
    
    # Enable debug prompt dumping
    os.environ["DEBUG_PROMPTS"] = "true"
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Print configuration
    print("\n" + "="*80)
    print("🧪 TALENT SUCCESS AGENT - TEST MODE")
    print("="*80)
    print(f"📋 Conversation ID: {args.conversation_id}")
    print(f"🔄 Dry Run: {'✅ Enabled' if args.dry_run else '⚠️  DISABLED (will write to Intercom!)'}")
    print(f"📚 Procedure ID: {args.procedure_id or 'None (will search)'}")
    print(f"📨 Messages: {len(args.messages)} message(s)")
    for i, msg in enumerate(args.messages, 1):
        preview = msg[:80] + "..." if len(msg) > 80 else msg
        print(f"   {i}. {preview}")
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
        # Build graph
        print("🔧 Building agent graph...")
        graph = build_graph()
        
        # Format messages
        formatted_messages = [
            {"role": "user", "content": msg}
            for msg in args.messages
        ]
        
        # Prepare initial state
        initial_state = {
            "conversation_id": args.conversation_id,
            "messages": formatted_messages,
            "mode": "test",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        # Add user details if provided
        if args.user_email or args.user_name:
            initial_state["user_details"] = {
                "email": args.user_email,
                "name": args.user_name
            }
            print(f"👤 Using provided user details:")
            if args.user_email:
                print(f"   Email: {args.user_email}")
            if args.user_name:
                print(f"   Name: {args.user_name}")
        
        # Add procedure_id if provided
        if args.procedure_id:
            initial_state["procedure_id"] = args.procedure_id
            print(f"🔍 Testing with procedure ID: {args.procedure_id}")
        
        # Run the agent
        print(f"🤖 Running agent in test mode...\n")
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
        print(f"\n❌ Failed to process test run")
        print(f"   Error: {e}")
        print(f"   Time: {elapsed_time:.2f}s\n")
        
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

