"""
Thread - The agent that connects the dots.
Simplified version for Hugging Face Spaces compatibility.
Last updated: 2025-06-06 19:53
"""

import os
import threading
from datetime import datetime
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from agent import ThreadAgent

# Load environment variables
load_dotenv()

# Background preload so the model loads before user input
def preload_model():
    from memory import MemoryManager
    _ = MemoryManager()  # force model load

threading.Thread(target=preload_model).start()

# Initialize the agent
agent = ThreadAgent()


def save_groq_key(api_key: str) -> str:
    """Save the Groq API key to environment variables."""
    if not api_key.strip():
        return "❌ **Error** - API key cannot be empty"
    try:
        os.environ["GROQ_API_KEY"] = api_key.strip()
        return "✅ **Success** - Groq API key saved (will be used on next message)"
    except Exception as e:
        return f"❌ **Error** - Failed to save API key: {str(e)}"


def check_api_key() -> str:
    """Check if Groq API key is configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return "✅ **Connected** - Groq API key is configured"
    else:
        return "❌ **Not Configured** - Please set your Groq API key"


def save_groq_key_and_clear(api_key: str) -> Tuple[str, str]:
    """Save the API key and clear the input field."""
    status = save_groq_key(api_key)
    if "Success" in status:
        success = agent.reload_groq_client()
        if success:
            status += " - Client reloaded successfully"
        else:
            status += " - Warning: Client reload failed"
    return "", status


async def process_message(message: str, history: List) -> Tuple[str, List, str]:
    """Process a user message and return updated interface elements."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please configure your Groq API key first!"})
        return "", history, get_initial_memory_panel()

    response, base_memory_content = await agent.process_message(message)
    similar_memories = agent.memory.retrieve_similar(message, top_k=3)
    memory_display = format_memory_panel(base_memory_content, similar_memories)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history, memory_display


def get_initial_memory_panel() -> str:
    """Get the initial memory panel content."""
    return format_memory_panel(agent._get_memory_panel(), [])


def clear_all_memory() -> str:
    """Clear all memory and return updated panel content."""
    base_content = agent.clear_memory()
    return format_memory_panel(base_content, [])


def show_memory_stats() -> str:
    """Show detailed memory statistics."""
    stats = agent.get_memory_stats()
    size_kb = stats["index_size_bytes"] / 1024

    latest_time = "None"
    if stats.get("latest_timestamp"):
        latest_time = stats["latest_timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return f"""### 📊 Memory Statistics
- Total Entries: {stats['total_entries']}
- Vector Index Size: {size_kb:.1f} KB
- Embedding Dimension: {stats['embedding_dim']}
- Latest Activity: {latest_time}

### 🔍 Similar Memories
*Start a conversation to see similar memories in action.*
"""


def format_memory_panel(base_content: str, similar_memories: List) -> str:
    """Format the memory panel with base content and similar memories."""
    memory_display = base_content + "\n\n### 🔍 Similar Memories\n"

    if similar_memories:
        for i, mem in enumerate(similar_memories, 1):
            text = mem["text"][:80] + ("..." if len(mem["text"]) > 80 else "")
            try:
                timestamp = datetime.fromisoformat(mem["timestamp"]).strftime("%m/%d %H:%M")
            except (ValueError, KeyError):
                timestamp = "Unknown"
            memory_display += f"**{i}.** {text}\n📊 Relevance: {mem['similarity']:.3f} | 📅 {timestamp}\n\n"
    else:
        memory_display += "*No relevant memories found.*\n"

    return memory_display


# Create the simplified Gradio interface
with gr.Blocks(title="Thread - Memory Agent") as app:
    gr.Markdown("""
    # 🧠 Thread - The Agent that Connects the Dots
    **Thread** is a memory-aware conversational agent powered by **GroqCloud**.
    It retrieves relevant past context and helps you think across conversations.
    """)

    # API Configuration Section
    with gr.Accordion("🔐 Configure API", open=True):
        with gr.Row():
            with gr.Column(scale=3):
                api_key_input = gr.Textbox(
                    label="Groq API Key",
                    placeholder="Enter your Groq API key here...",
                    type="password",
                    info="Get yours at https://console.groq.com/"
                )
            with gr.Column(scale=1):
                save_key_btn = gr.Button("💾 Save Key", variant="primary")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        
        api_status = gr.Markdown(value=check_api_key())

    # Main Interface
    with gr.Row():
        with gr.Column(scale=7):
            # Simple Chat Interface
            chatbot = gr.Chatbot(
                [],
                label="💬 Conversation",
                height=450,
                type="messages"
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    scale=9,
                    lines=1
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        # Memory Panel
        with gr.Column(scale=3):
            memory_panel = gr.Markdown(
                label="🧠 Memory Panel",
                value=get_initial_memory_panel()
            )
            with gr.Row():
                stats_btn = gr.Button("📊 Stats", variant="secondary")
                clear_btn = gr.Button("🗑️ Reset", variant="secondary")

    # Event handlers
    save_key_btn.click(
        save_groq_key_and_clear,
        inputs=[api_key_input],
        outputs=[api_key_input, api_status]
    )
    refresh_btn.click(check_api_key, outputs=[api_status])
    
    send_btn.click(
        process_message, 
        inputs=[msg_input, chatbot], 
        outputs=[msg_input, chatbot, memory_panel]
    )
    msg_input.submit(
        process_message, 
        inputs=[msg_input, chatbot], 
        outputs=[msg_input, chatbot, memory_panel]
    )
    
    stats_btn.click(show_memory_stats, outputs=[memory_panel])
    clear_btn.click(clear_all_memory, outputs=[memory_panel])


# Launch the app with minimal configuration
if __name__ == "__main__":
    print("🚀 Starting Thread application with GroqCloud integration...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )
