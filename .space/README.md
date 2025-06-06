# Thread Space Configuration

This Space runs the Thread agent, a memory-aware conversational AI powered by GroqCloud.

## Configuration

The Space is configured with:
- Python 3.11
- Gradio 4.44.0
- 16GB RAM
- No GPU required

## Environment Variables

Required:
- `GROQ_API_KEY`: Your GroqCloud API key

Automatically set:
- `GRADIO_SERVER_NAME`: 0.0.0.0
- `GRADIO_SERVER_PORT`: 7860
- `PYTHONUNBUFFERED`: 1

## Deployment Notes

1. The app runs on port 7860
2. Memory system uses FAISS for vector search
3. No persistent storage - memory resets on Space restart
4. API key must be set in Space settings 