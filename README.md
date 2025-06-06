# 🧠 Thread - The Agent that Connects the Dots

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GroqCloud](https://img.shields.io/badge/Powered%20by-GroqCloud-purple.svg)](https://console.groq.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Agents-MCP-Hackathon/thread-agent)
[![agent-demo-track](https://img.shields.io/badge/track-agent--demo-orange)](https://huggingface.co/spaces/Agents-MCP-Hackathon/thread-agent)

**Thread** is a memory-aware conversational AI agent that connects information across conversations. It features sophisticated semantic memory management, intelligent topic detection, and context-aware response generation powered by GroqCloud's Llama3-70B model.

## 🎥 Demo Video

[Coming Soon] Watch Thread in action, demonstrating its memory capabilities and topic detection.

## ✨ Features

### 🧠 **Semantic Memory System**
- **FAISS Vector Database**: High-performance similarity search using sentence embeddings
- **Persistent Context**: Remembers and connects information across conversations
- **Real-time Memory Display**: Shows relevant memories with similarity scores
- **Memory Analytics**: Detailed statistics and insights

### 🎯 **Intelligent Topic Detection**
- **Automatic Classification**: Detects conversation topics from 7 domains
- **Dynamic Tracking**: Updates topic focus as conversations evolve
- **Context-Aware Prompts**: Tailors responses based on detected topics

### 💡 **Smart Suggestions**
- **Next Step Recommendations**: AI-generated actionable suggestions
- **Domain-Specific Guidance**: Specialized advice for different topic areas
- **Progress Tracking**: Helps users advance their thinking

## 🚀 Getting Started

### 1. **Get Your GroqCloud API Key**
- Sign up at [console.groq.com](https://console.groq.com/)
- Create a new API key
- Copy your API key

### 2. **Configure Thread**
- Open the "🔐 Configure API" section
- Paste your Groq API key
- Click "💾 Save Key"
- Wait for the "✅ Success" message

### 3. **Start Conversing**
- Type your message in the chat input
- Watch the memory panel for relevant past context
- Observe topic detection and next step suggestions
- Use the memory panel to track conversation history

## 🧪 Example Conversations

Try these topics to see Thread's capabilities:

**Technical Discussion:**
```
"We're having issues with our Python code's performance"
"What debugging steps should we take?"
```

**Business Strategy:**
```
"How can we improve our market position?"
"What metrics should we track for growth?"
```

**Data Analysis:**
```
"I need to create a dashboard for our KPIs"
"How should we analyze this dataset?"
```

## 🔧 Memory Management

Use the memory panel controls:
- **📊 Stats**: View detailed memory statistics
- **🗑️ Reset**: Clear all memories to start fresh
- **Similar Memories**: See real-time relevant context

## 🛠️ Space Configuration

The Hugging Face Space is configured with:
- Python 3.11
- Gradio 4.44.0
- 16GB RAM
- No GPU required

### Environment Variables

Required:
- `GROQ_API_KEY`: Your GroqCloud API key

Automatically set:
- `GRADIO_SERVER_NAME`: 0.0.0.0
- `GRADIO_SERVER_PORT`: 7860
- `PYTHONUNBUFFERED`: 1

### Deployment Notes

1. The app runs on port 7860
2. Memory system uses FAISS for vector search
3. No persistent storage - memory resets on Space restart
4. API key must be set in Space settings

## 🤝 Contributing

This project is part of the Hugging Face Agents Hackathon. Feel free to:
- Submit issues for bugs or feature requests
- Fork and create pull requests
- Share your experience and suggestions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroqCloud](https://console.groq.com/) for providing fast LLM inference
- [FAISS](https://faiss.ai/) for efficient vector similarity search
- [SentenceTransformers](https://www.sbert.net/) for semantic embeddings
- [Gradio](https://gradio.app/) for the web interface framework

---

**Tags**: `agent-demo-track`, `conversational-ai`, `semantic-memory`, `groq`, `faiss`, `gradio` 