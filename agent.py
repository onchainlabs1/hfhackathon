"""
🤖 THREAD AGENT MODULE
Creative memory agent with LLM reasoning and planning separation
Aligned with Model Context Protocol (MCP) specifications
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

from memory import SemanticMemory

# Load environment variables
load_dotenv()


class ThreadAgent:
    """
    Creative memory agent that combines semantic memory with LLM reasoning.
    Tracks creative intent and provides proactive suggestions.
    """
    
    def __init__(self):
        self.memory = SemanticMemory()
        self.groq_client = None
        self.creative_intent = {}
        self.conversation_context = []
        
        print("🤖 ThreadAgent initialized")
    
    def _initialize_groq_client(self) -> bool:
        """Initialize Groq client with API key."""
        api_key = os.getenv("GROQ_API_KEY")
        print(f"🔑 API Key check: {'Found' if api_key else 'Not found'}")
        
        if not api_key or not api_key.strip():
            print("⚠️ GROQ_API_KEY not found in environment variables")
            self.groq_client = None
            return False
        
        try:
            print("📦 Importing Groq...")
            import groq
            print(f"📦 Groq library imported successfully")
            
            print("🔧 Creating Groq client instance...")
            # Absolutely minimal initialization
            self.groq_client = groq.Groq(api_key=api_key.strip())
            
            print("✅ Groq client created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating Groq client: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            print(f"🔍 Error details: {str(e)}")
            
            # Try alternative approach
            print("🔄 Trying alternative initialization...")
            try:
                from groq import Groq as GroqClient
                self.groq_client = GroqClient(api_key=api_key.strip())
                print("✅ Alternative initialization successful!")
                return True
            except Exception as e2:
                print(f"❌ Alternative approach also failed: {e2}")
                self.groq_client = None
                return False
    
    def reload_groq_client(self) -> bool:
        """Reload Groq client (useful after API key update)."""
        return self._initialize_groq_client()
    
    async def process_message(self, user_message: str) -> Tuple[str, str]:
        """
        Process user message and generate response with memory integration.
        
        Args:
            user_message: User's input message
            
        Returns:
            Tuple of (response, memory_panel_content)
        """
        if not user_message.strip():
            return "Please enter a message.", self._get_memory_panel()
        
        # 1. Retrieve similar memories BEFORE storing current message
        similar_memories = self.memory.retrieve_similar(user_message, top_k=3)
        
        # 2. Store user message in memory
        self.memory.add_memory(user_message, "user")
        
        # 3. Update creative intent tracking
        self._update_creative_intent(user_message, "user")
        
        # 4. Initialize Groq client if needed and generate response
        if not self.groq_client:
            self._initialize_groq_client()
        
        # Generate response using LLM reasoning
        if not self.groq_client:
            response = self._generate_fallback_response(user_message, similar_memories)
        else:
            response = await self._generate_response(user_message, similar_memories)
        
        # 5. Store assistant response in memory
        self.memory.add_memory(response, "assistant")
        
        # 6. Update creative intent with response
        self._update_creative_intent(response, "assistant")
        
        # 7. Get updated memory panel
        memory_panel = self._get_memory_panel(similar_memories)
        
        return response, memory_panel
    
    async def _generate_response(
        self, 
        user_message: str, 
        similar_memories: List[Dict]
    ) -> str:
        """
        Generate response using Groq LLM with memory context.
        
        Args:
            user_message: Current user message
            similar_memories: Retrieved similar memories
            
        Returns:
            Generated response
        """
        try:
            # Build system prompt with memory context
            system_prompt = self._build_system_prompt(similar_memories)
            
            # Get recent conversation context (last 5 turns)
            recent_context = self.memory.get_recent_context(limit=5)
            
            # Build messages for Groq
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent context
            for context in recent_context:
                messages.append({
                    "role": context["role"],
                    "content": context["content"]
                })
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add proactive suggestion
            suggestion = self._generate_proactive_suggestion(user_message, assistant_response)
            if suggestion:
                assistant_response += f"\n\n💡 {suggestion}"
            
            return assistant_response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return self._generate_fallback_response(user_message, similar_memories)
    
    def _build_system_prompt(self, similar_memories: List[Dict]) -> str:
        """
        Build enhanced system prompt with memory context and creative intent.
        
        Args:
            similar_memories: Retrieved similar memories
            
        Returns:
            Enhanced system prompt
        """
        base_prompt = """You are Thread, a creative memory agent that connects ideas across conversations. You have access to semantic memory that helps you maintain context and build upon previous discussions.

Key traits:
- Creative and insightful, helping users develop ideas
- Reference relevant past conversations when helpful
- Suggest creative next steps and connections
- Maintain continuity across conversations
- Be concise but thoughtful in responses"""

        # Add memory context if available
        if similar_memories:
            memory_context = "\n\nRELEVANT MEMORY CONTEXT:\n"
            for i, mem in enumerate(similar_memories, 1):
                similarity = mem.get('similarity', 0)
                role = mem.get('role', 'unknown')
                text = mem.get('text_preview', mem.get('text', ''))
                timestamp = mem.get('formatted_timestamp', 'unknown')
                
                memory_context += f"{i}. [{role}] {text} (relevance: {similarity:.3f}, {timestamp})\n"
            
            base_prompt += memory_context
        
        # Add creative intent context
        if self.creative_intent:
            intent_context = "\n\nCREATIVE INTENT TRACKING:\n"
            for key, value in self.creative_intent.items():
                if isinstance(value, (int, float)) and value > 0:
                    intent_context += f"- {key}: {value}\n"
            
            if len(intent_context) > len("\n\nCREATIVE INTENT TRACKING:\n"):
                base_prompt += intent_context
        
        base_prompt += "\n\nUse this context to provide relevant, creative responses that build upon previous conversations."
        
        return base_prompt
    
    def _generate_fallback_response(
        self, 
        user_message: str, 
        similar_memories: List[Dict]
    ) -> str:
        """
        Generate fallback response when Groq API is unavailable.
        
        Args:
            user_message: User's message
            similar_memories: Retrieved memories
            
        Returns:
            Fallback response
        """
        response = f"**🤖 Thread Agent Response**\n\n"
        response += f"📝 **Your Message:** {user_message}\n\n"
        response += "⚠️ **Status:** Creative reasoning is limited (API key needed for full functionality)\n\n"
        
        if similar_memories:
            response += "🧠 **Related Memories Found:**\n\n"
            for i, mem in enumerate(similar_memories, 1):
                text = mem.get('text_preview', '')
                similarity = mem.get('similarity', 0)
                timestamp = mem.get('formatted_timestamp', 'unknown')
                role = mem.get('role', 'unknown')
                role_emoji = "👤" if role == "user" else "🤖"
                
                response += f"**{i}.** {role_emoji} {text}\n"
                response += f"    📊 Relevance: **{similarity:.3f}** | 📅 {timestamp}\n\n"
        else:
            response += "🔍 **No similar memories found yet.** Continue the conversation to build context!\n\n"
        
        response += "---\n\n"
        response += "💡 **To unlock full creative capabilities:**\n"
        response += "1. Get your free API key from [console.groq.com](https://console.groq.com)\n"
        response += "2. Configure it in the 🔐 Groq API Configuration panel\n"
        response += "3. Enjoy enhanced responses with Llama3-70B!"
        
        return response
    
    def _update_creative_intent(self, message: str, role: str) -> None:
        """
        Update creative intent tracking based on message content.
        
        Args:
            message: The message to analyze
            role: "user" or "assistant"
        """
        message_lower = message.lower()
        
        # Creative writing indicators
        writing_keywords = ['write', 'story', 'novel', 'script', 'poem', 'blog', 'article']
        if any(keyword in message_lower for keyword in writing_keywords):
            self.creative_intent['writing_projects'] = self.creative_intent.get('writing_projects', 0) + 1
        
        # Planning indicators
        planning_keywords = ['plan', 'strategy', 'goal', 'roadmap', 'timeline', 'project']
        if any(keyword in message_lower for keyword in planning_keywords):
            self.creative_intent['planning_activities'] = self.creative_intent.get('planning_activities', 0) + 1
        
        # Learning indicators
        learning_keywords = ['learn', 'understand', 'explain', 'how', 'what', 'why']
        if any(keyword in message_lower for keyword in learning_keywords):
            self.creative_intent['learning_queries'] = self.creative_intent.get('learning_queries', 0) + 1
        
        # Problem-solving indicators
        problem_keywords = ['problem', 'issue', 'challenge', 'solve', 'fix', 'debug']
        if any(keyword in message_lower for keyword in problem_keywords):
            self.creative_intent['problem_solving'] = self.creative_intent.get('problem_solving', 0) + 1
    
    def _generate_proactive_suggestion(self, user_message: str, response: str) -> Optional[str]:
        """
        Generate proactive suggestions based on conversation context.
        
        Args:
            user_message: User's message
            response: Generated response
            
        Returns:
            Proactive suggestion or None
        """
        message_lower = user_message.lower()
        
        # Writing project suggestions
        if any(word in message_lower for word in ['story', 'novel', 'script']):
            return "Would you like me to help you develop characters, plot outline, or setting details?"
        
        # Video/content suggestions
        if any(word in message_lower for word in ['video', 'youtube', 'content']):
            return "Should we brainstorm a content calendar or video script outline?"
        
        # Learning suggestions
        if any(word in message_lower for word in ['learn', 'study', 'understand']):
            return "Would you like me to create a learning roadmap or suggest practice exercises?"
        
        # Planning suggestions
        if any(word in message_lower for word in ['plan', 'project', 'goal']):
            return "Shall we break this down into actionable steps with timelines?"
        
        # Creative suggestions based on intent history
        if self.creative_intent.get('writing_projects', 0) > 2:
            return "I notice you're working on writing projects. Would you like to explore a new creative direction?"
        
        return None
    
    def _get_memory_panel(self, similar_memories: Optional[List[Dict]] = None) -> str:
        """
        Generate memory panel content for UI display.
        
        Args:
            similar_memories: Optional pre-retrieved memories
            
        Returns:
            Formatted memory panel content
        """
        stats = self.memory.get_stats()
        
        panel = f"""### 📝 Project Memory
**Total Entries:** {stats['total_entries']}
**Memory Size:** {stats['index_size_bytes'] / 1024:.1f} KB
**Embedding Dim:** {stats['embedding_dim']}
"""
        
        if stats['latest_timestamp']:
            latest = stats['latest_timestamp'].strftime("%m/%d %H:%M")
            panel += f"**Last Update:** {latest}\n"
        
        # Add creative intent summary
        if self.creative_intent:
            panel += "\n### 🎯 Creative Intent\n"
            for intent, count in self.creative_intent.items():
                if count > 0:
                    panel += f"- {intent.replace('_', ' ').title()}: {count}\n"
        
        # Add similar memories section
        panel += "\n### 🔍 Similar Memories\n"
        
        if similar_memories:
            for i, mem in enumerate(similar_memories, 1):
                text = mem.get('text_preview', '')
                similarity = mem.get('similarity', 0)
                timestamp = mem.get('formatted_timestamp', '')
                role = mem.get('role', 'unknown')
                
                # Role emoji
                role_emoji = "👤" if role == "user" else "🤖"
                
                panel += f"**{i}.** {role_emoji} {text}\n"
                panel += f"   📊 Relevance: {similarity:.3f} | 📅 {timestamp}\n\n"
        else:
            panel += "*No relevant memories found. Start a conversation to build context.*\n"
        
        return panel
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics."""
        base_stats = self.memory.get_stats()
        base_stats.update({
            "creative_intent": self.creative_intent,
            "groq_connected": self.groq_client is not None,
            "total_conversations": len([m for m in self.memory.memories if m.role == "user"])
        })
        return base_stats
    
    def clear_memory(self) -> str:
        """Clear all memories and reset agent state."""
        self.memory.reset()
        self.creative_intent.clear()
        self.conversation_context.clear()
        return self._get_memory_panel()


# MCP-aligned reasoning endpoints (placeholder for future MCP integration)
class MCPReasoningServer:
    """Placeholder MCP server endpoints for reasoning operations."""
    
    def __init__(self, agent: ThreadAgent):
        self.agent = agent
    
    async def reasoning_endpoint(self, request: Dict) -> Dict:
        """MCP /reasoning endpoint"""
        message = request.get("message", "")
        context = request.get("context", [])
        
        # This would integrate with the agent's reasoning pipeline
        return {
            "reasoning": "Enhanced reasoning with memory context",
            "confidence": 0.85,
            "creative_intent": self.agent.creative_intent
        }
    
    async def planning_endpoint(self, request: Dict) -> Dict:
        """MCP /planning endpoint"""
        goal = request.get("goal", "")
        constraints = request.get("constraints", [])
        
        # This would integrate with the agent's planning capabilities
        return {
            "plan": "Step-by-step plan based on memory and context",
            "timeline": "Suggested timeline",
            "next_actions": ["Action 1", "Action 2"]
        }
