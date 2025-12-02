# Draft Node Procedure Injection

## Summary

Added procedure injection to the draft node to match the pattern used in plan and coverage nodes. Previously, procedures selected by the procedure node were not being passed to the draft prompt, meaning the LLM generating the final response (Melvin) didn't have access to internal procedure guidance.

This ensures consistent procedure handling across all nodes - Plan, Coverage, and Draft all now receive and use procedure guidance when available.

## Changes Made

### 1. Import Added (Line 11)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
from src.utils.prompts import build_conversation_and_user_context, format_procedure_for_prompt
```

Added `format_procedure_for_prompt` import to use the same procedure formatting function as plan and coverage nodes.

### 2. Extract Procedure from State (Lines 67-68)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
# Get selected procedure from state
selected_procedure = state.get("selected_procedure")
```

Extract the `selected_procedure` from the state (populated by the procedure node if a matching procedure was found).

### 3. Pass Procedure to _generate_response (Lines 72-81)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
response = _generate_response(
    formatted_context["conversation_history"],
    formatted_context["user_details"],
    tool_data,
    docs_data,
    coverage_reasoning,
    validation_feedback,
    mode,
    selected_procedure  # ← NEW PARAMETER
)
```

### 4. Update _generate_response Signature (Lines 130-139)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
def _generate_response(
    conversation_history: str,
    user_details: str,
    tool_data: Dict[str, Any],
    docs_data: Dict[str, Any],
    coverage_reasoning: str = None,
    validation_feedback: Dict[str, Any] = None,
    mode: str = None,
    selected_procedure: Dict[str, Any] = None  # ← NEW PARAMETER
) -> Dict[str, Any]:
```

Added `selected_procedure` parameter with proper type hints and documentation.

### 5. Pass Procedure to _create_system_prompt (Line 159)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
system_prompt = _create_system_prompt(
    conversation_history, 
    user_details, 
    context_data, 
    coverage_reasoning, 
    validation_feedback, 
    mode, 
    selected_procedure  # ← NEW PARAMETER
)
```

### 6. Update _create_system_prompt Signature (Line 290)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
def _create_system_prompt(
    conversation_history: str, 
    user_details: str, 
    context_data: Dict[str, Any], 
    coverage_reasoning: str = None, 
    validation_feedback: Dict[str, Any] = None, 
    mode: str = None, 
    selected_procedure: Dict[str, Any] = None  # ← NEW PARAMETER
) -> str:
```

### 7. Format and Inject Procedure (Lines 341-363)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
# Format procedure if available
procedure_text = format_procedure_for_prompt(selected_procedure)

# Select prompt based on mode
if mode == "splvin":
    prompt_name = PROMPT_NAMES["DRAFT_NODE_SLACK"]
    print(f"🎯 Using splvin draft prompt: {prompt_name}")
else:
    prompt_name = PROMPT_NAMES["DRAFT_NODE"]

# Get prompt from LangSmith
prompt_template = get_prompt(prompt_name)

# Format the prompt with variables
prompt = prompt_template.format(
    conversation_history=conversation_history,
    user_details=user_details,
    data_summary=full_data_summary,
    procedure=procedure_text  # ← NEW VARIABLE PASSED TO TEMPLATE
)
```

The `format_procedure_for_prompt()` function formats the procedure with proper headers and instructions:
- Returns empty string if no procedure
- Otherwise formats with title, reasoning, content, and important instructions

### 8. Update Debug Metadata (Line 371)

**File:** `src/ts_agent/nodes/draft/draft.py`

```python
metadata = {
    "Prompt Length": f"{len(prompt)} characters",
    "Has Validation Feedback": bool(validation_feedback),
    "Has Coverage Reasoning": bool(coverage_reasoning),
    "Has Procedure": bool(selected_procedure),  # ← NEW METADATA
    "Tool Data Count": len(context_data.get('tool_data', {})),
    "Docs Data Count": len(context_data.get('docs_data', {})),
    "Documentation Content Count": len(context_data.get('documentation_content', []))
}
```

## Behavior

### Before
- Procedure selected by procedure node was stored in state
- Plan and Coverage nodes could see and use the procedure
- **Draft node did NOT see the procedure**
- Final response didn't follow procedure guidance

### After
- Procedure selected by procedure node is stored in state
- Plan, Coverage, **and Draft** nodes all see and use the procedure
- Final response follows procedure guidance
- Consistent behavior across all nodes

## Flow Diagram

```
State: selected_procedure
    ↓
draft_node()
    ↓ (extracts from state)
_generate_response(selected_procedure)
    ↓
_create_system_prompt(selected_procedure)
    ↓
format_procedure_for_prompt(selected_procedure)
    ↓ (returns formatted text)
prompt_template.format(procedure=procedure_text)
    ↓
LLM sees procedure in prompt
```

## Testing

Updated `test_draft_procedure_injection.py` (renamed from test_splvin_draft_prompt.py) to include procedure testing:
- Added sample `selected_procedure` dict
- Passes procedure to `_generate_response()`
- Verifies procedure appears in debug dump
- Focuses on standard Melvin mode

To test:
```bash
cd talent-success-agent
DEBUG_PROMPTS=true python test_draft_procedure_injection.py
# Check debug_prompts/draft_prompt_*.txt for procedure section
```

## Important Note: LangSmith Prompt Template

The actual prompt templates are stored in LangSmith, not in the local `.txt` files. The LangSmith prompt for `talent-success-agent-draft` **must include the `{procedure}` placeholder** for this to work.

If the LangSmith prompt doesn't have `{procedure}`, the `.format()` call will raise a KeyError.

To verify/update LangSmith prompt:
1. Go to LangSmith dashboard
2. Find prompt: `talent-success-agent-draft`
3. Ensure it has `{procedure}` placeholder in the appropriate location
4. Suggested location: After user details, before data summary (matching plan node pattern)

Example placement:
```
**USER DETAILS:**
{user_details}

{procedure}

**RELEVANT INFORMATION:**
{data_summary}
```

## Related Files

- `src/ts_agent/nodes/plan/plan.py` - Plan node (already had procedure injection)
- `src/ts_agent/nodes/coverage/coverage.py` - Coverage node (already had procedure injection)
- `src/utils/prompts.py` - Contains `format_procedure_for_prompt()` function
- `test_splvin_draft_prompt.py` - Updated test file

