"""
Utility functions for text processing and formatting.
"""

from typing import Dict


def format_chat_message(role: str, content: str) -> Dict[str, str]:
    """
    Format a message for the chat interface.
    
    Args:
        role: The role of the message sender (user/assistant)
        content: The message content
        
    Returns:
        Dictionary containing role and content for chat interface
    """
    return {"role": role, "content": content} 