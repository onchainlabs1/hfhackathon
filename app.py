"""
🚀 THREAD - THE CREATIVE MEMORY AGENT
Gradio-based agentic app aligned with Model Context Protocol (MCP)
Main application interface with semantic memory and LLM reasoning
"""

import os
import asyncio
from typing import List, Tuple, Optional
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

from agent import ThreadAgent
from memory import SemanticMemory

# Load environment variables
load_dotenv()

# Global agent instance
agent = ThreadAgent()


def format_message_for_chatbot(role: str, content: str) -> dict:
    """Format message for Gradio Chatbot component."""
    return {"role": role, "content": content}


async def process_chat_message(
    message: str, 
    history: List[dict]
) -> Tuple[List[dict], str, str]:
    """
    Process user message and update chat history.
    
    Args:
        message: User's input message
        history: Current chat history
        
    Returns:
        Updated history, empty input field, memory panel content
    """
    if not message.strip():
        return history, "", agent._get_memory_panel()
    
    # Add user message to history
    history.append(format_message_for_chatbot("user", message))
    
    # Process message through agent
    response, memory_panel = await agent.process_message(message)
    
    # Add assistant response to history
    history.append(format_message_for_chatbot("assistant", response))
    
    return history, "", memory_panel


def get_memory_stats() -> str:
    """Get comprehensive memory statistics."""
    stats = agent.get_memory_stats()
    
    stats_text = f"""### 📊 **Memory Statistics**

**📈 Overview:**
- Total Entries: {stats['total_entries']}
- Memory Size: {stats['index_size_bytes'] / 1024:.1f} KB
- Embedding Dimension: {stats['embedding_dim']}
- Conversations: {stats['total_conversations']}

**🔌 System Status:**
- Groq API: {'✅ Connected' if stats['groq_connected'] else '❌ Not Connected'}
- Memory System: ✅ Active
- MCP Aligned: ✅ Ready

**🎯 Creative Intent:**"""
    
    if stats['creative_intent']:
        for intent, count in stats['creative_intent'].items():
            if count > 0:
                stats_text += f"\n- {intent.replace('_', ' ').title()}: {count}"
    else:
        stats_text += "\n- No creative patterns detected yet"
    
    if stats['latest_timestamp']:
        latest = stats['latest_timestamp'].strftime("%m/%d/%Y %H:%M:%S")
        stats_text += f"\n\n**⏰ Last Activity:** {latest}"
    
    return stats_text


def clear_memory() -> Tuple[List[dict], str]:
    """Clear all memories and reset chat."""
    memory_panel = agent.clear_memory()
    return [], memory_panel


def refresh_memory_panel() -> str:
    """Refresh the memory panel display."""
    return agent._get_memory_panel()


def save_api_key(api_key: str) -> str:
    """
    Save Groq API key to environment (memory only, no file write).
    
    Args:
        api_key: The Groq API key
        
    Returns:
        Status message
    """
    if not api_key.strip():
        return "❌ Please enter a valid API key"
    
    # Set environment variable (memory only)
    os.environ["GROQ_API_KEY"] = api_key.strip()
    
    # Note: Groq client will be initialized when processing messages (not here)
    return "✅ API key saved successfully! Client will initialize on first message."


def get_api_status() -> str:
    """Get current API connection status."""
    if agent.groq_client:
        return "🟢 **Status:** Connected to Groq API"
    else:
        return "🔴 **Status:** Groq API not connected. Please configure your API key below."


# MCP-aligned placeholder endpoints (for future MCP server integration)


# MCP-aligned placeholder endpoints (for future MCP server integration)
async def mcp_context_endpoint():
    """Placeholder MCP /context endpoint"""
    context = agent.memory.get_recent_context()
    return {
        "context": context,
        "total_memories": agent.memory.total_entries,
        "timestamp": datetime.now().isoformat()
    }


async def mcp_memory_add_endpoint(text: str, role: str = "user"):
    """Placeholder MCP /memory/add endpoint"""
    agent.memory.add_memory(text, role)
    return {"success": True, "message": "Memory added successfully"}


async def mcp_memory_search_endpoint(query: str, top_k: int = 3):
    """Placeholder MCP /memory/search endpoint"""
    results = agent.memory.retrieve_similar(query, top_k)
    return {"results": results, "query": query, "count": len(results)}


# Create the Gradio interface directly
custom_css = """
.memory-panel {
    height: 400px;
    overflow-y: auto;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #dee2e6;
}

.stats-panel {
    height: 300px;
    overflow-y: auto;
    padding: 15px;
    background-color: #f1f3f4;
    border-radius: 8px;
    border: 1px solid #dadce0;
}

.api-status {
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 10px;
}
"""

with gr.Blocks(
    title="Thread - Creative Memory Agent",
    css=custom_css,
    theme=gr.themes.Soft()
) as app:
    
    # Header
    gr.Markdown("""
    # 🧠 **Thread – The Creative Memory Agent**
    *Agentic app with semantic memory • Model Context Protocol aligned • Powered by GroqCloud + Llama3-70B*
    """)
    
    with gr.Row():
        # Left Column - Chat Interface
        with gr.Column(scale=2):
            # Main Chat Interface
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                type="messages",
                height=500,
                show_copy_button=True,
                show_share_button=False
            )
            
            # Input Row
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Start a creative conversation...",
                    lines=2,
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            # Action Buttons Row
            with gr.Row():
                stats_btn = gr.Button("📊 Stats", variant="secondary")
                reset_btn = gr.Button("🗑️ Reset", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        
        # Right Column - Memory & Configuration
        with gr.Column(scale=1):
            # Memory Panel
            memory_panel = gr.Markdown(
                value=agent._get_memory_panel(),
                label="📝 Project Memory",
                elem_classes=["memory-panel"]
            )
            
            # API Configuration Accordion
            with gr.Accordion("🔐 Groq API Configuration", open=False) as api_accordion:
                api_status = gr.Markdown(value=get_api_status(), elem_classes=["api-status"])
                
                gr.Markdown("""
                **Configure your Groq API key for full creative reasoning:**
                1. Get your free API key from [console.groq.com](https://console.groq.com)
                2. Enter it below and click Save
                3. The key is stored in memory only (not saved to disk)
                """)
                
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        type="password",
                        placeholder="gsk_...",
                        scale=3
                    )
                    save_key_btn = gr.Button("💾 Save", variant="primary", scale=1)
            
            # Stats Panel (initially hidden)
            stats_panel = gr.Markdown(
                visible=False,
                elem_classes=["stats-panel"]
            )
    
    # Event Handlers
    
    # Chat message processing
    async def handle_message(message, history):
        return await process_chat_message(message, history)
    
    # Send button and Enter key
    send_btn.click(
        fn=handle_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, memory_panel],
        api_name="chat"
    )
    
    msg_input.submit(
        fn=handle_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, memory_panel]
    )
    
    # Stats button - toggle visibility
    def toggle_stats():
        current_visible = stats_panel.visible
        if not current_visible:
            stats_content = get_memory_stats()
            return gr.Markdown(value=stats_content, visible=True)
        else:
            return gr.Markdown(visible=False)
    
    stats_btn.click(
        fn=toggle_stats,
        outputs=[stats_panel]
    )
    
    # Reset button
    reset_btn.click(
        fn=clear_memory,
        outputs=[chatbot, memory_panel]
    )
    
    # Refresh button
    refresh_btn.click(
        fn=refresh_memory_panel,
        outputs=[memory_panel]
    )
    
    # API key save
    save_key_btn.click(
        fn=save_api_key,
        inputs=[api_key_input],
        outputs=[api_status]
    )
    
    # Footer
    gr.Markdown("""
    ---
    **🏆 Built for Hugging Face Agents Hackathon** | 
    [GitHub](https://github.com/onchainlabs1/hfhackathon) | 
    Powered by SentenceTransformers + FAISS + GroqCloud
    """)

# Launch configuration
if __name__ == "__main__":
    app.launch(server_port=7860, show_error=True) 