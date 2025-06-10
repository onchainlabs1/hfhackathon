"""
🚀 THREAD - HUGGING FACE SPACES OPTIMIZED VERSION
Simplified deployment version for HF Spaces with faster loading
"""

import os
import asyncio
from typing import List, Tuple
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import our modules
from agent import ThreadAgent

# Initialize agent
print("🚀 Initializing Thread Agent for HF Spaces...")
agent = ThreadAgent()

async def process_chat_message(message: str, history: List[dict]) -> Tuple[List[dict], str, str]:
    """Process user message and update chat history."""
    if not message.strip():
        return history, "", agent._get_memory_panel()
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    
    # Process message through agent
    response, memory_panel = await agent.process_message(message)
    
    # Add assistant response to history
    history.append({"role": "assistant", "content": response})
    
    return history, "", memory_panel

def save_api_key(api_key: str) -> str:
    """Save Groq API key to environment."""
    if not api_key.strip():
        return "❌ Please enter a valid API key"
    
    os.environ["GROQ_API_KEY"] = api_key.strip()
    success = agent.reload_groq_client()
    
    if success:
        return "✅ API key saved and Groq client initialized successfully!"
    else:
        return "⚠️ API key saved but client initialization failed. Check logs for details."

def get_memory_stats() -> str:
    """Get memory statistics."""
    stats = agent.get_memory_stats()
    return f"""### 📊 Memory Statistics
- **Total Entries:** {stats['total_entries']}
- **Memory Size:** {stats['index_size_bytes'] / 1024:.1f} KB
- **Groq API:** {'✅ Connected' if stats['groq_connected'] else '❌ Not Connected'}
- **Conversations:** {stats['total_conversations']}"""

def clear_memory() -> Tuple[List[dict], str]:
    """Clear all memories and reset chat."""
    memory_panel = agent.clear_memory()
    return [], memory_panel

# Create the Gradio interface
with gr.Blocks(
    title="Thread - Creative Memory Agent",
    theme=gr.themes.Soft(),
    css="""
    .memory-panel {
        background-color: #1f2937;
        color: #f9fafb;
        padding: 15px;
        border-radius: 8px;
        height: 400px;
        overflow-y: auto;
    }
    """
) as app:
    
    gr.Markdown("""
    # 🧠 **Thread – The Creative Memory Agent**
    *Semantic memory + LLM reasoning + MCP-aligned architecture*
    
    **Built for Hugging Face Agents Hackathon 2024**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chat Interface
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                type="messages",
                height=500,
                value=[]
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Start a creative conversation...",
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            with gr.Row():
                stats_btn = gr.Button("📊 Stats", variant="secondary")
                reset_btn = gr.Button("🗑️ Reset", variant="secondary")
        
        with gr.Column(scale=1):
            # Memory Panel
            memory_panel = gr.Markdown(
                value=agent._get_memory_panel(),
                label="📝 Project Memory",
                elem_classes=["memory-panel"]
            )
            
            # API Configuration
            with gr.Accordion("🔐 Groq API Configuration", open=False):
                gr.Markdown("""
                **To enable full creative capabilities:**
                1. Get your free API key from [console.groq.com](https://console.groq.com)
                2. Enter it below and click Save
                """)
                
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        type="password",
                        placeholder="gsk_...",
                        scale=3
                    )
                    save_key_btn = gr.Button("💾 Save", variant="primary", scale=1)
                
                api_status = gr.Markdown(
                    value="🔴 **Status:** Groq API not connected"
                )
    
    # Event handlers
    send_btn.click(
        fn=process_chat_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, memory_panel]
    )
    
    msg_input.submit(
        fn=process_chat_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, memory_panel]
    )
    
    stats_btn.click(
        fn=get_memory_stats,
        outputs=[memory_panel]
    )
    
    reset_btn.click(
        fn=clear_memory,
        outputs=[chatbot, memory_panel]
    )
    
    save_key_btn.click(
        fn=save_api_key,
        inputs=[api_key_input],
        outputs=[api_status]
    )
    
    gr.Markdown("""
    ---
    **🏆 Hugging Face Agents Hackathon 2024** | 
    [GitHub](https://github.com/onchainlabs1/hfhackathon) | 
    Powered by SentenceTransformers + FAISS + GroqCloud
    """)

# Launch for HF Spaces
if __name__ == "__main__":
    print("🚀 Launching Thread Agent on HF Spaces...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 