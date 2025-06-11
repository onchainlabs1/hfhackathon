---
title: Thread Agent
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - agent-demo-track
  - memory
  - conversational-ai
  - groq
  - faiss
---

# 🧠 Thread - The Agent that Connects the Dots

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GroqCloud](https://img.shields.io/badge/Powered%20by-GroqCloud-purple.svg)](https://console.groq.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Agents-MCP-Hackathon/thread-agent)
[![agent-demo-track](https://img.shields.io/badge/track-agent--demo-orange)](https://huggingface.co/spaces/Agents-MCP-Hackathon/thread-agent)

**Thread** is a memory-aware conversational agent that bridges past conversations with present context. Built for the Hugging Face Agents Hackathon, Thread demonstrates sophisticated memory management through semantic similarity search and vector embeddings.

## 🎥 Demo Video

🎬 **[Watch Thread Agent in Action](https://www.youtube.com/watch?v=HZ-62Hy7Xbw)** 

See Thread Agent demonstrating its memory capabilities and conversation continuity features in this comprehensive walkthrough.

## 🎯 **Core Features**

### 🧠 **Semantic Memory System**
- **Vector-based memory** using FAISS + SentenceTransformers
- **Contextual retrieval** of relevant past conversations
- **Persistent knowledge** across conversations
- **Real-time similarity scoring** for memory relevance

### 💬 **Intelligent Conversation**
- **GroqCloud integration** for fast, high-quality responses
- **Context-aware responses** informed by memory
- **Topic detection** and conversation flow tracking
- **Suggested next steps** based on conversation patterns

### 📊 **Memory Analytics**
- **Live memory statistics** and usage tracking
- **Vector index monitoring** and performance metrics
- **Conversation history** with relevance scoring
- **Memory management** tools (reset, statistics)

## 🚀 **Quick Start**

### 1. **Get Your Groq API Key**
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
- Observe real-time memory retrieval and relevance scoring
- Use the memory panel to track conversation history

## 🧪 **Example Conversations**

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

## 🔧 **Memory Management**

Use the memory panel controls:
- **📊 Stats**: View detailed memory statistics
- **🗑️ Clear**: Reset all memories to start fresh
- **Similar Memories**: See real-time relevant context

## 🛠️ **Technical Architecture**

### **Memory System**
- **SentenceTransformers**: `all-MiniLM-L6-v2` for text embeddings
- **FAISS**: Vector similarity search with L2 distance
- **384-dimensional vectors** for semantic representation
- **Jaccard similarity** for memory relevance scoring

### **AI Integration**
- **GroqCloud API**: Fast inference with Llama3-70B model
- **Context window**: 8K tokens with memory-informed prompts
- **Response generation**: Temperature 0.7 for balanced creativity
- **Error handling**: Graceful fallbacks and user feedback

## 🎯 **Hackathon Track: Agent Demo**

Thread is submitted under the **agent-demo-track** for the Hugging Face Agents Hackathon. It showcases:

- **Advanced memory management** with vector embeddings
- **Semantic search** for contextual conversation
- **Real-time analytics** and memory visualization
- **Production-ready deployment** on Hugging Face Spaces

## 🔗 **Links**

- **Live Demo**: [Thread Agent Space](https://huggingface.co/spaces/Agents-MCP-Hackathon/thread-agent)
- **Source Code**: Available in this repository
- **Hackathon**: [Hugging Face Agents Hackathon](https://huggingface.co/blog/agents-hackathon)

## 🤝 **Contributing**

This project is part of the Hugging Face Agents Hackathon. Feel free to:
- Submit issues for bugs or feature requests
- Fork and create pull requests
- Share your experience and suggestions

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for the Hugging Face Agents Hackathon*

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