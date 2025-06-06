"""
Core agent logic for the Thread system.
Handles conversation flow, response generation, and memory integration.
"""

import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

from memory import MemoryManager
from utils import format_chat_message

# Load environment variables
load_dotenv()


class ThreadAgent:
    """
    Main agent class that handles conversation and memory integration.

    This class manages the conversational AI agent with features including:
    - Topic detection and tracking
    - Context-aware response generation
    - Memory integration with semantic search
    - Next step suggestions
    """

    def __init__(self) -> None:
        """Initialize the Thread agent with memory and conversation tracking."""
        self.memory = MemoryManager()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_topic = "General Discussion"
        self.suggested_next_step = "Start a conversation to explore topics"

        # Initialize Groq client once for reuse
        self.groq_client = self._initialize_groq_client()

        self.base_system_prompt = """You are Thread, an intelligent agent that connects information
        and maintains context through sophisticated memory management. You should:
        1. Be concise but informative in your responses
        2. Reference relevant past information when appropriate
        3. Maintain a professional and helpful tone
        4. Ask clarifying questions when needed
        5. Offer actionable next steps when relevant"""

        # Topic keywords for classification
        self.topic_keywords = {
            "pricing": [
                "price",
                "cost",
                "budget",
                "expensive",
                "cheap",
                "pricing",
                "fee",
            ],
            "technical": [
                "code",
                "programming",
                "bug",
                "error",
                "development",
                "software",
            ],
            "business": [
                "strategy",
                "revenue",
                "growth",
                "market",
                "customer",
                "sales",
            ],
            "data_analysis": [
                "data",
                "analytics",
                "metrics",
                "dashboard",
                "report",
                "insights",
            ],
            "project_management": [
                "project",
                "deadline",
                "timeline",
                "milestone",
                "task",
                "planning",
            ],
            "customer_support": [
                "support",
                "help",
                "issue",
                "problem",
                "ticket",
                "complaint",
            ],
            "machine_learning": [
                "ml",
                "ai",
                "model",
                "training",
                "prediction",
                "algorithm",
            ],
        }

    def _initialize_groq_client(self) -> Optional[Groq]:
        """
        Initialize Groq client once during initialization.

        Returns:
            Groq client instance or None if not configured
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            return None
        try:
            return Groq(api_key=api_key)
        except Exception:
            return None

    def reload_groq_client(self) -> bool:
        """
        Reload Groq client from current environment variables.
        
        This method should be called after updating the GROQ_API_KEY
        environment variable to apply the new API key.
        
        Returns:
            True if client was successfully reloaded, False otherwise
        """
        self.groq_client = self._initialize_groq_client()
        return self.groq_client is not None

    def _detect_topic(self, message: str) -> str:
        """
        Detect conversation topic based on keywords in the message.

        Args:
            message: The user message to analyze

        Returns:
            The detected topic or current topic if no match found
        """
        message_lower = message.lower()

        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                topic_scores[topic] = score

        # Return the topic with highest score, or keep current if no clear match
        if topic_scores:
            best_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            return best_topic.replace("_", " ").title()

        return self.current_topic

    def _generate_next_step_suggestion(self, message: str, topic: str) -> str:
        """
        Generate intelligent next step suggestions based on topic and context.

        Args:
            message: The user message
            topic: The detected topic

        Returns:
            Suggested next step for the user
        """
        message_lower = message.lower()

        # Topic-specific suggestions
        if "pricing" in topic.lower():
            if any(word in message_lower for word in ["compare", "options", "plans"]):
                return "Consider reviewing pricing tiers and value propositions"
            return "Analyze cost-benefit scenarios or competitor pricing"

        elif "technical" in topic.lower():
            if any(word in message_lower for word in ["bug", "error", "issue"]):
                return "Document the issue and create a troubleshooting plan"
            return "Review technical documentation or consider code optimization"

        elif "business" in topic.lower():
            if any(word in message_lower for word in ["strategy", "growth"]):
                return "Define KPIs and create actionable business metrics"
            return "Schedule stakeholder review or market analysis"

        elif "data" in topic.lower():
            if any(word in message_lower for word in ["report", "dashboard"]):
                return "Set up automated reporting or data visualization"
            return "Validate data sources and define analysis methodology"

        elif "project" in topic.lower():
            if any(word in message_lower for word in ["deadline", "timeline"]):
                return "Create detailed project timeline with milestones"
            return "Review project scope and resource allocation"

        elif "support" in topic.lower():
            return "Create knowledge base entry or escalation procedure"

        elif "machine learning" in topic.lower():
            if any(word in message_lower for word in ["model", "training"]):
                return "Evaluate model performance and consider optimization"
            return "Gather more training data or explore feature engineering"

        # Default suggestions based on conversation patterns
        if "?" in message:
            return "Research the topic further or consult domain experts"
        elif any(word in message_lower for word in ["help", "how", "what"]):
            return "Break down the problem into smaller, actionable steps"
        else:
            return "Continue exploring this topic or move to related areas"

    def _build_enhanced_system_prompt(self, memory_context: str, topic: str) -> str:
        """
        Build a rich, context-aware system prompt.

        Args:
            memory_context: Relevant memories context
            topic: Current conversation topic

        Returns:
            Enhanced system prompt with context
        """
        enhanced_prompt = f"""{self.base_system_prompt}

CURRENT CONTEXT:
- Topic: {topic}
- Memory Context: {memory_context}

INSTRUCTIONS:
- Reference past information when it adds value to your response
- Justify your suggestions with reasoning from previous conversations
- Offer specific, actionable next steps when relevant
- Connect current discussion to related topics from memory
- If discussing {topic.lower()}, focus on domain-specific insights and best practices

Remember: Your goal is to create meaningful connections between ideas and help users progress their thinking."""

        return enhanced_prompt

    async def process_message(self, message: str) -> Tuple[str, str]:
        """
        Process a user message and generate a response.

        Args:
            message: The user's input message

        Returns:
            Tuple of (response, memory_panel_content)
        """
        # Update current topic based on user message
        self.current_topic = self._detect_topic(message)

        # Generate next step suggestion
        self.suggested_next_step = self._generate_next_step_suggestion(
            message, self.current_topic
        )

        # Retrieve relevant memories BEFORE adding current message to prevent self-matching
        relevant_memories = self.memory.retrieve_similar(message, top_k=3)
        memory_context = self._format_memory_context(relevant_memories)

        # Add user message to conversation history
        self.conversation_history.append(format_chat_message("user", message))

        # Store message in memory AFTER retrieval
        self.memory.add_entry(message, source="user")

        try:
            # Generate response using GroqCloud
            response = await self._generate_response(message, memory_context)

            # Add response to conversation history and memory
            self.conversation_history.append(format_chat_message("assistant", response))
            self.memory.add_entry(response, source="assistant")

            # Get updated memory panel content
            memory_panel = self._get_memory_panel()

            return response, memory_panel

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            return error_msg, self._get_memory_panel()

    async def _generate_response(self, message: str, memory_context: str) -> str:
        """
        Generate a response using GroqCloud with enhanced context.

        Args:
            message: The user message
            memory_context: Relevant memory context

        Returns:
            Generated response from the AI model
        """
        groq_client = self.groq_client
        if not groq_client:
            return self._generate_placeholder_response(message)

        # Build enhanced system prompt
        enhanced_system_prompt = self._build_enhanced_system_prompt(
            memory_context, self.current_topic
        )

        messages = [{"role": "system", "content": enhanced_system_prompt}]

        # Add recent conversation history (last 5 messages)
        messages.extend(self.conversation_history[-5:])

        # Add current user message
        messages.append({"role": "user", "content": message})

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192", messages=messages, temperature=0.7, max_tokens=1000
        )

        return response.choices[0].message.content

    def _generate_placeholder_response(self, message: str) -> str:
        """
        Generate a placeholder response when GroqCloud API is not configured.

        Args:
            message: The user message

        Returns:
            Placeholder response indicating API key needed
        """
        return (
            f"I understand you said: {message}\n\n"
            "This is a placeholder response since GROQ_API_KEY is not configured. "
            "Please set GROQ_API_KEY in your environment to enable full functionality."
        )

    def _format_memory_context(self, memories: List[Dict]) -> str:
        """
        Format memories into a context string for the agent.

        Args:
            memories: List of relevant memory entries

        Returns:
            Formatted memory context string
        """
        if not memories:
            return "No relevant memories found."

        context_parts = ["Relevant memories:"]
        for memory in memories:
            similarity = memory.get("similarity", 0)
            context_parts.append(
                f"- [{memory['source']}] {memory['text']} "
                f"(similarity: {similarity:.2f})"
            )
        return "\n".join(context_parts)

    def _get_memory_panel(self) -> str:
        """
        Get formatted memory panel content.

        Returns:
            Formatted memory statistics and information
        """
        stats = self.memory.get_stats()
        size_mb = stats["index_size_bytes"] / (1024 * 1024)

        latest_time = "None"
        if stats.get("latest_timestamp"):
            latest_time = stats["latest_timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"### 📊 Memory Statistics\n"
            f"- Total Entries: {stats['total_entries']}\n"
            f"- Vector Index Size: {size_mb:.2f} MB\n"
            f"- Embedding Dimension: {stats['embedding_dim']}\n"
            f"- Latest Entry: {latest_time}"
        )

    def clear_memory(self) -> str:
        """
        Clear all memories and return updated panel content.

        Returns:
            Updated memory panel content after clearing
        """
        self.memory.reset()
        self.conversation_history.clear()
        return self._get_memory_panel()

    def get_memory_stats(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dictionary containing memory statistics
        """
        return self.memory.get_stats()

    def is_api_configured(self) -> bool:
        """
        Check if Groq API key is configured.

        Returns:
            True if API key is configured and valid, False otherwise
        """
        groq_client = self.groq_client
        return groq_client is not None
