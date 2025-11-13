"""
Utility functions for formatting prompts with conversation history and user details.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


def format_conversation_history(messages: List[Dict[str, Any]], subject: Optional[str] = None) -> str:
    """
    Format conversation history for LLM prompts.
    
    Args:
        messages: List of conversation messages with role, content, and optional attachments
        subject: Optional conversation subject/title
        
    Returns:
        Formatted conversation history string including attachment information
    """
    parts = []
    
    # Add subject if available
    if subject:
        parts.append(f"Subject: {subject}\n")
    
    # Add messages
    if messages:
        parts.append("Conversation:")
        for i, message in enumerate(messages, 1):
            role = message.get("role", "unknown").title()
            content = message.get("content", "")
            parts.append(f"{i}. {role}: {content}")
            
            # Add attachment information if present
            attachments = message.get("attachments", [])
            if attachments:
                for j, attachment in enumerate(attachments, 1):
                    att_name = attachment.get("name", "Unknown file")
                    att_type = attachment.get("content_type", "unknown")
                    att_url = attachment.get("url", "")
                    
                    # Format attachment info for LLM
                    att_info = f"   📎 Attachment {j}: {att_name} (Type: {att_type})"
                    
                    # Include URL so LLM knows it's accessible
                    if att_url:
                        att_info += f"\n      URL: {att_url}"
                    
                    # Add file size if available
                    if "filesize" in attachment:
                        filesize_kb = attachment["filesize"] / 1024
                        att_info += f"\n      Size: {filesize_kb:.1f} KB"
                    
                    # Add dimensions for images
                    if "width" in attachment and "height" in attachment:
                        att_info += f"\n      Dimensions: {attachment['width']}x{attachment['height']}"
                    
                    parts.append(att_info)
    else:
        parts.append("Conversation: No messages available")
    
    return "\n".join(parts)


def format_user_details(user_details: Optional[Dict[str, Any]] = None, name: Optional[str] = None, email: Optional[str] = None) -> str:
    """
    Format user details for LLM prompts.
    
    Dynamically formats all fields in user_details dict. If user_details is not provided,
    falls back to legacy name/email parameters for backward compatibility.
    
    Args:
        user_details: Dictionary of user details (any fields)
        name: User's name (legacy, for backward compatibility)
        email: User's email (legacy, for backward compatibility)
        
    Returns:
        Formatted user details string with all available fields
    """
    parts = []
    
    # If user_details dict is provided, format all fields dynamically
    if user_details:
        for key, value in user_details.items():
            if value:  # Only include non-empty values
                # Format the key nicely (e.g., "user_id" -> "User ID")
                formatted_key = key.replace("_", " ").title()
                parts.append(f"{formatted_key}: {value}")
    
    # Fallback to legacy parameters if user_details not provided
    elif name or email:
        if name:
            parts.append(f"Name: {name}")
        if email:
            parts.append(f"Email: {email}")
        
    if not parts:
        parts.append("User details: Not available")
    
    return "\n".join(parts)


def build_conversation_and_user_context(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Build formatted conversation history and user details from state.
    
    Args:
        state: Agent state containing messages, subject, and user_details
        
    Returns:
        Dictionary with formatted conversation_history and user_details strings
        
    Raises:
        ValueError: If both messages and subject are missing from state
    """
    messages = state.get("messages", [])
    subject = state.get("subject")
    
    # Validate: Either messages or subject must be present
    has_messages = messages and len(messages) > 0
    has_subject = subject and subject.strip()
    
    if not has_messages and not has_subject:
        raise ValueError("No conversation messages or subject found in state")
    
    # Pass entire user_details dict for dynamic formatting
    user_details = state.get("user_details", {})
    
    return {
        "conversation_history": format_conversation_history(messages, subject),
        "user_details": format_user_details(user_details=user_details)
    }


def format_procedure_for_prompt(selected_procedure: Optional[Dict[str, Any]] = None) -> str:
    """
    Format procedure for injection into LLM prompts.
    
    Returns empty string if no procedure, otherwise formats with context.
    This function is used by both plan and coverage nodes.
    
    Args:
        selected_procedure: Optional selected procedure dict from state
        
    Returns:
        Formatted procedure text or empty string
    """
    if not selected_procedure:
        return ""
    
    title = selected_procedure.get("title", "Untitled Procedure")
    content = selected_procedure.get("content", "")
    reasoning = selected_procedure.get("reasoning", "")
    
    procedure_text = "**INTERNAL PROCEDURE GUIDANCE:**\n"
    procedure_text += f"An internal procedure has been identified that provides REQUIRED guidance for this scenario:\n\n"
    procedure_text += f"Title: {title}\n\n"
    
    if reasoning:
        procedure_text += f"Why this procedure was selected: {reasoning}\n\n"
    
    procedure_text += f"Procedure Details:\n{content}\n\n"
    procedure_text += "**IMPORTANT: Follow the procedure's steps and use ALL required tools listed in 'Tools Required'. The procedure provides mandatory guidance for handling this scenario correctly.**"
    
    return procedure_text


def convert_messages_to_langchain_with_vision(
    messages: List[Dict[str, Any]], 
    subject: Optional[str] = None,
    user_name: Optional[str] = None,
    user_email: Optional[str] = None
) -> List[BaseMessage]:
    """
    Convert conversation messages to LangChain message format with vision support.
    
    This function creates structured messages that allow the LLM to actually process
    images via URLs, not just be aware of them.
    
    Args:
        messages: List of conversation messages with role, content, and optional attachments
        subject: Optional conversation subject to prepend
        user_name: Optional user name to prepend
        user_email: Optional user email to prepend
        
    Returns:
        List of LangChain messages (HumanMessage/AIMessage) with image_url content for attachments
        
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Look at this", "attachments": [{"url": "...", "content_type": "image/png"}]}
        ... ]
        >>> lc_messages = convert_messages_to_langchain_with_vision(messages)
        >>> # Can now be used with: llm.invoke(lc_messages)
    """
    langchain_messages = []
    
    # Add context as a system-like message (using HumanMessage as first message)
    context_parts = []
    if subject:
        context_parts.append(f"Subject: {subject}")
    if user_name or user_email:
        context_parts.append("User Details:")
        if user_name:
            context_parts.append(f"  Name: {user_name}")
        if user_email:
            context_parts.append(f"  Email: {user_email}")
    
    if context_parts:
        # Prepend context to first message or create a separate context message
        context_text = "\n".join(context_parts) + "\n\n---\n\n"
    else:
        context_text = ""
    
    # Convert each message
    for i, message in enumerate(messages):
        role = message.get("role", "user")
        content_text = message.get("content", "")
        attachments = message.get("attachments", [])
        
        # Prepend context to first message
        if i == 0 and context_text:
            content_text = context_text + content_text
        
        # Check if there are image attachments
        has_images = any(
            att.get("content_type", "").startswith("image/")
            for att in attachments
        )
        
        if has_images:
            # Create structured content with text and images
            content_array = []
            
            # Add text content
            if content_text:
                content_array.append({
                    "type": "text",
                    "text": content_text
                })
            
            # Add each image attachment
            for attachment in attachments:
                content_type = attachment.get("content_type", "")
                if content_type.startswith("image/"):
                    url = attachment.get("url", "")
                    if url:
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                                "detail": "auto"  # Can be "low", "high", or "auto"
                            }
                        })
            
            # Create message with structured content
            if role == "assistant":
                # Note: AIMessage doesn't support image content, so we keep it as text
                # Images in assistant messages are rare anyway
                langchain_messages.append(AIMessage(content=content_text))
            else:
                langchain_messages.append(HumanMessage(content=content_array))
        else:
            # Regular text-only message
            # Include non-image attachment info in text
            if attachments:
                attachment_info = []
                for j, att in enumerate(attachments, 1):
                    att_name = att.get("name", "Unknown file")
                    att_type = att.get("content_type", "unknown")
                    att_url = att.get("url", "")
                    attachment_info.append(
                        f"📎 Attachment {j}: {att_name} (Type: {att_type}, URL: {att_url})"
                    )
                if attachment_info:
                    content_text = content_text + "\n\n" + "\n".join(attachment_info)
            
            if role == "assistant":
                langchain_messages.append(AIMessage(content=content_text))
            else:
                langchain_messages.append(HumanMessage(content=content_text))
    
    return langchain_messages
