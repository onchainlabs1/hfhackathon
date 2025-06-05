# Thread - The Agent that Connects the Dots

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GroqCloud](https://img.shields.io/badge/Powered%20by-GroqCloud-purple.svg)](https://console.groq.com/)

**Thread** is a memory-aware conversational AI agent that connects information across conversations. It features sophisticated semantic memory management, intelligent topic detection, and context-aware response generation powered by GroqCloud's Llama3-70B model.

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

### 🖥️ **Modern UI**
- **Gradio Interface**: Clean, responsive web interface
- **Live Updates**: Real-time memory and topic displays
- **Professional Design**: Modern gradients and intuitive layout
- **API Management**: Easy GroqCloud API key configuration

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- GroqCloud API key ([Get yours here](https://console.groq.com/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/onchainlabs1/hfhackathon.git
cd hfhackathon
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the application**
```bash
python app.py
```

6. **Open your browser**
Navigate to `http://localhost:7862` to start using Thread!

## 🏗️ Architecture

### Core Components

- **`agent.py`**: Main conversational agent with topic detection and memory integration
- **`memory.py`**: FAISS-based vector memory system with semantic search
- **`app.py`**: Gradio web interface with real-time updates
- **`utils.py`**: Utility functions for text processing and formatting

### Technology Stack

- **AI Model**: GroqCloud Llama3-70B-8192
- **Vector Search**: FAISS with SentenceTransformers
- **Embeddings**: all-MiniLM-L6-v2 (384-dimensional)
- **Web Interface**: Gradio 4.44.0
- **Environment**: Python 3.8+ with asyncio support

## 📊 Usage Examples

### Topic Detection
Thread automatically detects and tracks conversation topics:
- **Technical**: "We have a bug in our Python code"
- **Business**: "How can we improve our sales strategy?"
- **Data Analysis**: "I need to create a dashboard for our metrics"

### Memory Integration
Ask follow-up questions and Thread will connect relevant past information:
- Previous discussions about similar topics
- Related context from earlier conversations
- Semantic connections between ideas

### Smart Suggestions
Get actionable next steps based on your conversation:
- Domain-specific recommendations
- Context-aware guidance
- Progress-oriented suggestions

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your GroqCloud API key (required)

### Customization
- Modify topic keywords in `agent.py` for custom domains
- Adjust memory similarity thresholds in `memory.py`
- Customize UI styling in `app.py`

## 📸 Screenshots

### Main Interface
*A screenshot showing the main chat interface with memory panel*

### Topic Detection
*A screenshot showing real-time topic detection and suggestions*

### Memory Visualization
*A screenshot showing the memory panel with similar memories*

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroqCloud](https://console.groq.com/) for providing fast LLM inference
- [FAISS](https://faiss.ai/) for efficient vector similarity search
- [SentenceTransformers](https://www.sbert.net/) for semantic embeddings
- [Gradio](https://gradio.app/) for the web interface framework

## 📞 Support

- Create an issue for bug reports or feature requests
- Join our community discussions
- Check out the documentation for detailed usage guides

---

**Thread** - Connecting the dots in conversational AI 🧠✨ 