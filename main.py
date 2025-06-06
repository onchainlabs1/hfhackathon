"""
Thread - Memory-aware conversational agent
Simple, reliable version for Hugging Face Spaces
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

# Background model preload
def preload_model():
    from memory import MemoryManager
    _ = MemoryManager()

threading.Thread(target=preload_model).start()

# Initialize agent
agent = ThreadAgent()

def save_groq_key(api_key: str) -> str:
    if not api_key.strip():
        return "❌ API key cannot be empty"
    try:
        os.environ["GROQ_API_KEY"] = api_key.strip()
        return "✅ Groq API key saved successfully"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def check_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return "✅ Groq API key is configured"
    else:
        return "❌ Please set your Groq API key"

def save_key_and_clear(api_key: str) -> Tuple[str, str]:
    status = save_groq_key(api_key)
    if "Success" in status:
        agent.reload_groq_client()
    return "", status

async def chat_fn(message: str, history: List) -> Tuple[str, List, str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please configure your Groq API key first!"})
        return "", history, get_memory_panel()

    response, memory_content = await agent.process_message(message)
    similar_memories = agent.memory.retrieve_similar(message, top_k=3)
    memory_display = format_memory(memory_content, similar_memories)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return "", history, memory_display

def get_memory_panel() -> str:
    return format_memory(agent._get_memory_panel(), [])

def clear_memory() -> str:
    base_content = agent.clear_memory()
    return format_memory(base_content, [])

def show_stats() -> str:
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

def format_memory(base_content: str, similar_memories: List) -> str:
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

# Create Gradio interface
with gr.Blocks(title="Thread Agent") as demo:
    gr.Markdown("""
    # 🧠 Thread - The Agent that Connects the Dots
    Memory-aware conversational agent powered by GroqCloud.
    """)

    # API Configuration
    with gr.Accordion("🔐 Configure API", open=True):
        with gr.Row():
            with gr.Column(scale=3):
                api_input = gr.Textbox(
                    label="Groq API Key",
                    placeholder="Enter your Groq API key...",
                    type="password",
                    info="Get yours at https://console.groq.com/"
                )
            with gr.Column(scale=1):
                save_btn = gr.Button("💾 Save", variant="primary")
                refresh_btn = gr.Button("🔄 Check", variant="secondary")
        
        status = gr.Markdown(value=check_api_key())

    # Main Interface
    with gr.Row():
        with gr.Column(scale=7):
            # Chat Interface - Fixed configuration
            chat = gr.Chatbot(
                value=[],
                label="💬 Conversation", 
                height=450,
                type="messages"
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message...",
                    scale=9
                )
                send = gr.Button("Send", scale=1, variant="primary")

        # Memory Panel
        with gr.Column(scale=3):
            memory = gr.Markdown(
                label="🧠 Memory",
                value=get_memory_panel()
            )
            with gr.Row():
                stats_btn = gr.Button("📊 Stats", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    # Event handlers
    save_btn.click(save_key_and_clear, inputs=[api_input], outputs=[api_input, status])
    refresh_btn.click(check_api_key, outputs=[status])
    send.click(chat_fn, inputs=[msg, chat], outputs=[msg, chat, memory])
    msg.submit(chat_fn, inputs=[msg, chat], outputs=[msg, chat, memory])
    stats_btn.click(show_stats, outputs=[memory])
    clear_btn.click(clear_memory, outputs=[memory])

# Launch with simple configuration
if __name__ == "__main__":
    print("🚀 Starting Thread application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 