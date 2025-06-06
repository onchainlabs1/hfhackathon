"""
Configuration settings for the Thread application.
"""

import os

# Gradio configuration
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    "share": not bool(os.getenv("SPACE_ID")),  # Only share when not in Spaces
    "debug": False,
    "show_error": True,
    "root_path": "",
    "ssl_verify": False,
    "show_api": False,  # Disable API docs for better performance
    "quiet": True  # Reduce logging noise
}

# Memory configuration
MEMORY_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_similar": 3,
    "max_tokens": 8000
}

# Agent configuration
AGENT_CONFIG = {
    "model": "llama3-70b-8192",
    "temperature": 0.7,
    "max_tokens": 1000
} 