#!/usr/bin/env python3
"""Direct test of draft prompt with procedure injection."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["DEBUG_PROMPTS"] = "true"

from ts_agent.nodes.draft.draft import _generate_response

def main():
    """Test the draft prompt with procedure injection."""
    
    print("\n" + "="*80)
    print("🧪 TESTING DRAFT PROMPT WITH PROCEDURE INJECTION")
    print("="*80)
    print("Directly calling draft generation with procedure")
    print("="*80 + "\n")
    
    # Test data
    conversation_history = """Subject: Project Help Request

Conversation:
1. User: I need help understanding my application status"""
    
    user_details = """Name: Test User
Email: test@example.com
User ID: user-12345-test"""
    
    tool_data = {
        "get_user_applications": [{
            "text": '{"applications": [{"id": "app_123", "position": "Software Engineer", "status": "interview_scheduled", "company": "Tech Corp"}]}'
        }]
    }
    
    docs_data = {}
    
    coverage_reasoning = "We have application information. Ready to draft response."
    
    # Test procedure data - simulates what procedure node would provide
    selected_procedure = {
        "title": "Handling Application Status Inquiries",
        "content": "1. Review the user's current applications\n2. Explain the current status clearly\n3. Provide next steps if applicable\n4. Link to relevant help documentation",
        "reasoning": "This procedure was selected because the user asked about application status"
    }
    
    print("📝 Generating response for Melvin (standard mode)...")
    print("🎯 Should use 'talent-success-agent-draft' prompt")
    print("🔧 Including procedure in prompt")
    print()
    
    # Generate response with procedure (standard Melvin mode)
    response = _generate_response(
        conversation_history=conversation_history,
        user_details=user_details,
        tool_data=tool_data,
        docs_data=docs_data,
        coverage_reasoning=coverage_reasoning,
        validation_feedback=None,
        mode=None,  # Standard mode (not splvin)
        selected_procedure=selected_procedure
    )
    
    print("\n" + "="*80)
    print("✅ RESPONSE GENERATED")
    print("="*80)
    print(f"Response Type: {response['response_type']}")
    print(f"Response:\n{response['response']}")
    print("\n" + "="*80)
    print("📂 Check debug_prompts/draft_prompt_*.txt to verify:")
    print("   ✓ Procedure section should be present")
    print("   ✓ Should include procedure title and content")
    print("   ✓ Should include procedure guidance instructions")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
