"""
Thread - The agent that connects the dots.
Main application entry point with improved Gradio interface using GroqCloud API.
"""

import os
from datetime import datetime
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from agent import ThreadAgent

# Load environment variables
load_dotenv()

# Initialize the agent
agent = ThreadAgent()


def save_groq_key(api_key: str) -> str:
    """
    Save the Groq API key to environment variables.

    Args:
        api_key: The Groq API key to save

    Returns:
        Status message indicating success or failure
    """
    if not api_key.strip():
        return "❌ **Error** - API key cannot be empty"
    try:
        os.environ["GROQ_API_KEY"] = api_key.strip()
        return "✅ **Success** - Groq API key saved (will be used on next message)"
    except Exception as e:
        return f"❌ **Error** - Failed to save API key: {str(e)}"


def check_api_key() -> str:
    """
    Check if Groq API key is configured.

    Returns:
        Status message indicating API key configuration status
    """
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return "✅ **Connected** - Groq API key is configured"
    else:
        return "❌ **Not Configured** - Please set your Groq API key"


def save_groq_key_and_clear(api_key: str) -> Tuple[str, str]:
    """
    Save the API key and clear the input field.

    Args:
        api_key: The API key to save

    Returns:
        Tuple of (empty_string, status_message)
    """
    status = save_groq_key(api_key)

    # Reload the Groq client to apply the new API key
    if "Success" in status:
        success = agent.reload_groq_client()
        if success:
            status += " - Client reloaded successfully"
        else:
            status += " - Warning: Client reload failed"

    return "", status


async def process_message(
    message: str, history: List
) -> Tuple[str, List, str, str, str]:
    """
    Process a user message and return updated interface elements.

    Args:
        message: The user's input message
        history: Current chat history

    Returns:
        Tuple of (empty_input, updated_history, memory_display, topic, next_step)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please configure your Groq API key first!"})
        return (
            "",
            history,
            get_initial_memory_panel(),
            format_topic_display(agent.current_topic),
            format_next_step_display(agent.suggested_next_step),
        )

    response, base_memory_content = await agent.process_message(message)
    similar_memories = agent.memory.retrieve_similar(message, top_k=3)
    memory_display = format_memory_panel(base_memory_content, similar_memories)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return (
        "",
        history,
        memory_display,
        format_topic_display(agent.current_topic),
        format_next_step_display(agent.suggested_next_step),
    )


def get_initial_memory_panel() -> str:
    """
    Get the initial memory panel content.

    Returns:
        Formatted memory panel content
    """
    return format_memory_panel(agent._get_memory_panel(), [])


def clear_all_memory() -> str:
    """
    Clear all memory and return updated panel content.

    Returns:
        Updated memory panel content after clearing
    """
    base_content = agent.clear_memory()
    return format_memory_panel(base_content, [])


def show_memory_stats() -> str:
    """
    Show detailed memory statistics.

    Returns:
        Formatted memory statistics
    """
    stats = agent.get_memory_stats()
    size_kb = stats["index_size_bytes"] / 1024

    latest_time = "None"
    if stats.get("latest_timestamp"):
        latest_time = stats["latest_timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return f"""### 📊 Detailed Memory Statistics

**Storage Info:**
- Total Entries: {stats['total_entries']}
- Vector Index Size: {size_kb:.1f} KB
- Embedding Dimension: {stats['embedding_dim']}

**Latest Activity:**
- Last Update: {latest_time}

### 🔍 Similar Memories
*Start a conversation to see similar memories in action.*
"""


def format_memory_panel(base_content: str, similar_memories: List) -> str:
    """
    Format the memory panel with base content and similar memories.

    Args:
        base_content: Base memory statistics content
        similar_memories: List of similar memory entries

    Returns:
        Formatted memory panel content
    """
    memory_display = base_content + "\n\n### 🔍 Similar Memories\n"

    if similar_memories:
        for i, mem in enumerate(similar_memories, 1):
            text = mem["text"][:80] + ("..." if len(mem["text"]) > 80 else "")
            try:
                timestamp = datetime.fromisoformat(mem["timestamp"]).strftime(
                    "%m/%d %H:%M"
                )
            except (ValueError, KeyError):
                timestamp = "Unknown"
            memory_display += (
                f"**{i}.** {text}\n"
                f"📊 Relevance: {mem['similarity']:.3f} | 📅 {timestamp}\n\n"
            )
    else:
        memory_display += "*No relevant memories found.*\n"

    return memory_display


def format_topic_display(topic: str) -> str:
    """
    Format the current topic for display with styled background.

    Args:
        topic: The current topic string

    Returns:
        HTML formatted topic display
    """
    return f"""
<div style="padding: 10px; background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); border-radius: 8px; margin-bottom: 10px;">
    <span style="color: white; font-weight: bold;">🎯 Current Topic: {topic}</span>
</div>
"""


def format_next_step_display(next_step: str) -> str:
    """
    Format the suggested next step for display with styled background.

    Args:
        next_step: The suggested next step string

    Returns:
        HTML formatted next step display
    """
    return f"""
<div style="padding: 10px; background: linear-gradient(90deg, #059669 0%, #0D9488 100%); border-radius: 8px; margin-bottom: 10px;">
    <span style="color: white; font-weight: bold;">💡 Suggested Next Step:</span><br/>
    <span style="color: white;">{next_step}</span>
</div>
"""


# Configure Gradio theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_label_background_fill="*primary_100",
    button_primary_background_fill="*primary_600"
)

# Create the Gradio interface
with gr.Blocks(
    theme=theme,
    title="Thread - The Agent that Connects the Dots"
) as app:
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
        
        api_status = gr.Markdown(
            value=check_api_key(),
            elem_classes=[
                "status-positive" if os.getenv("GROQ_API_KEY") else "status-negative"
            ]
        )

    # Main Interface
    with gr.Row():
        with gr.Column(scale=7):
            # Agent Intelligence Display
            with gr.Row():
                with gr.Column(scale=1):
                    current_topic_display = gr.Markdown(
                        value=format_topic_display(agent.current_topic),
                        elem_classes=["topic-display"]
                    )
                with gr.Column(scale=1):
                    next_step_display = gr.Markdown(
                        value=format_next_step_display(agent.suggested_next_step),
                        elem_classes=["next-step-display"]
                    )
            
            # Chat Interface
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=500,
                show_copy_button=True,
                type="messages"  # ✅ Required for Gradio 5 and Hugging Face Spaces
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    scale=9,
                    lines=1,
                    container=False
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        # Memory Panel
        with gr.Column(scale=3):
            memory_panel = gr.Markdown(
                label="🧠 Memory Panel",
                value=get_initial_memory_panel(),
                elem_classes=["memory-panel"]
            )
            with gr.Row():
                stats_btn = gr.Button("📊 Stats", variant="secondary")
                clear_btn = gr.Button("🗑️ Reset", variant="secondary")

    gr.Markdown("""
    ---
    💡 **Tips**:
    - Save your Groq API key to enable AI responses
    - The memory panel will display relevant memories after each input
    """)

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
        outputs=[
            msg_input, chatbot, memory_panel, 
            current_topic_display, next_step_display
        ]
    )
    msg_input.submit(
        process_message, 
        inputs=[msg_input, chatbot], 
        outputs=[
            msg_input, chatbot, memory_panel, 
            current_topic_display, next_step_display
        ]
    )
    
    stats_btn.click(show_memory_stats, outputs=[memory_panel])
    clear_btn.click(clear_all_memory, outputs=[memory_panel])


# Launch the app
if __name__ == "__main__":
    print("🚀 Starting Thread application with GroqCloud integration...")
    try:
        from config import GRADIO_CONFIG
        app.launch(**GRADIO_CONFIG)
        print("✅ Application started successfully!")
    except Exception as e:
        print(f"❌ Error starting application: {str(e)}")
        print("💡 Tip: Try changing the port if it is already in use.")
