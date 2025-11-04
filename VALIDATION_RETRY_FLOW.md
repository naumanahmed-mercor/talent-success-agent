# Validation Retry Flow

## Overview
This document describes the validation retry mechanism that gives the draft node one more attempt to fix validation issues before escalating.

## Flow Diagram

```
Draft Node
    ↓
    → Generate response
    ↓
Validate Node (Attempt 1)
    ↓
    ├─ PASS → Response Node → Finalize
    │
    └─ FAIL (retry_count = 0)
        ↓
        → Add validation results to Intercom as note
        → Store validation feedback in state
        → Set retry_count = 1
        ↓
Draft Node (Retry)
    ↓
    → Generate response WITH validation feedback
    ↓
Validate Node (Attempt 2)
    ↓
    ├─ PASS → Response Node → Finalize
    │
    └─ FAIL (retry_count = 1)
        ↓
        → Escalate (validation failed after 2 attempts)
```

## Key Changes

### 1. State Schema (`types.py`)
- Changed `validate` field from `Dict` to `List[Dict]` to store **all validation attempts as an array**
- Added `max_validation_retries` parameter (default: 1) to control retry limit
- No separate `validation_feedback` field needed - draft reads directly from the `validate` array

### 2. Validate Schema (`validate/schemas.py`)
- Added `retry_count` field to track validation attempts
- Updated `next_action` to support `"draft"` as a valid action

### 3. Validate Node (`validate/validate.py`)
- **Stores validation data as an array** - appends each validation attempt instead of overwriting
- Uses configurable `max_validation_retries` from state (defaults to 1 if not set)
- Determines retry count from length of validation array: `len(existing_validations)`
- Checks `if current_retry_count < max_validation_retries` to decide if retry is allowed
- Dynamic logging shows attempt number: `📊 Validation attempt X/Y`
- On failure with retries remaining:
  - Routes back to draft
  - **Appends validation attempt to array** (draft will read from this)
- On final failure (no retries left):
  - Escalates with dynamic reason including attempt count
  - **Appends final validation attempt to array (all preserved)**

### 4. Draft Node (`draft/draft.py`)
- Checks latest entry in `validate` array to detect retry
- If `latest_validation.next_action == "draft"`, reads `validation_response` as feedback
- Passes validation feedback to the LLM prompt
- No cleanup needed - single source of truth in the array

### 5. Graph (`graph.py`)
- Updated `route_from_validate` to support routing back to draft
- Added `"draft": "draft"` edge from validate node

## Benefits

1. **Self-correction**: Draft gets one chance to fix validation issues automatically
2. **Detailed feedback**: Validation results are provided to the LLM for context
3. **Complete audit trail**: All validation attempts are stored in an array and logged as notes in Intercom
4. **No data loss**: Previous validation results are preserved when retrying
5. **Controlled retry**: Only one retry attempt to avoid infinite loops
6. **Graceful escalation**: If retry fails, escalate with clear context from both validation attempts

## Example Scenario

1. User asks a question
2. Agent gathers info and generates draft response
3. Validation fails (e.g., policy violation detected)
4. Validation note added to Intercom
5. Draft re-runs with validation feedback explaining the issue
6. Draft generates corrected response
7. Validation passes
8. Response sent to user

If validation fails again on step 7, the agent escalates instead.

## Data Structure

The `validate` field in state is now an array of validation attempts:

```python
state["validate"] = [
    {
        "validation_response": {...},  # Raw validation API response
        "overall_passed": False,
        "validation_note_added": True,
        "escalation_reason": None,
        "next_action": "draft",
        "retry_count": 0,
        "timestamp": "..."
    },
    {
        "validation_response": {...},  # Second validation attempt
        "overall_passed": False,
        "validation_note_added": True,
        "escalation_reason": "Validation failed after 2 attempts",
        "next_action": "escalate",
        "retry_count": 1,
        "timestamp": "..."
    }
]
```

This ensures:
- Complete history of all validation attempts
- Easy debugging and auditing
- No data loss when retrying
- Clear tracking of attempt number via array length

## Configuration

The retry behavior can be easily tuned via the `max_validation_retries` state parameter:

```python
# Default behavior (1 retry = 2 total attempts)
state["max_validation_retries"] = 1  # Default

# No retries (1 total attempt)
state["max_validation_retries"] = 0

# Two retries (3 total attempts)
state["max_validation_retries"] = 2
```

**Total attempts = max_validation_retries + 1**

This parameter can be set:
- At initialization in the initialize node
- Per conversation based on user tier/type
- Globally via environment configuration
- Dynamically based on conversation complexity



