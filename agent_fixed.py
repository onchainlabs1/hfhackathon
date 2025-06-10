"""
🤖 THREAD AGENT - ULTRA ROBUST VERSION
Zero cache conflicts, bulletproof Groq initialization
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from memory import SemanticMemory

# Load environment variables
load_dotenv()

class ThreadAgent:
    """Ultra-robust creative memory agent."""
    
    def __init__(self):
        self.memory = SemanticMemory()
        self.groq_client = None
        self.creative_intent = {}
        self.conversation_context = []
        print("🤖 ThreadAgent initialized")
    
    def _create_groq_client(self, api_key: str):
        """Create Groq client with zero cache conflicts."""
        try:
            # Method 1: Direct import and create
            exec("import groq")
            groq_module = sys.modules['groq']
            client = groq_module.Groq(api_key=api_key)
            return client, groq_module.__version__
        except Exception as e1:
            try:
                # Method 2: Importlib approach
                import importlib
                groq_module = importlib.import_module('groq')
                client = groq_module.Groq(api_key=api_key)
                return client, groq_module.__version__
            except Exception as e2:
                try:
                    # Method 3: __import__ approach
                    groq_module = __import__('groq')
                    client = groq_module.Groq(api_key=api_key)
                    return client, groq_module.__version__
                except Exception as e3:
                    print(f"❌ All Groq methods failed: {e1}, {e2}, {e3}")
                    return None, "unknown"
    
    def _initialize_groq_client(self) -> bool:
        """Ultra-robust Groq initialization."""
        api_key = os.getenv("GROQ_API_KEY")
        print(f"🔑 API Key check: {'Found' if api_key else 'Not found'}")
        
        if not api_key or not api_key.strip():
            print("⚠️ GROQ_API_KEY not found")
            self.groq_client = None
            return False
        
        print("🔧 Creating Groq client (ultra-robust method)...")
        
        # Clear any cached imports
        groq_modules = [k for k in sys.modules.keys() if 'groq' in k.lower()]
        for mod in groq_modules:
            if mod in sys.modules:
                del sys.modules[mod]
        
        client, version = self._create_groq_client(api_key.strip())
        
        if client:
            self.groq_client = client
            print(f"✅ Groq client created! Version: {version}")
            return True
        else:
            print("❌ All Groq initialization methods failed")
            self.groq_client = None
            return False
    
    def reload_groq_client(self) -> bool:
        """Reload Groq client."""
        return self._initialize_groq_client()
    
    async def process_message(self, user_message: str) -> Tuple[str, str]:
        """Process user message and generate response."""
        if not user_message.strip():
            return "Please enter a message.", self._get_memory_panel()
        
        # Retrieve similar memories
        similar_memories = self.memory.retrieve_similar(user_message, top_k=3)
        
        # Store user message
        self.memory.add_memory(user_message, "user")
        
        # Initialize Groq if needed
        if not self.groq_client:
            self._initialize_groq_client()
        
        # Generate response
        if not self.groq_client:
            response = self._generate_fallback_response(user_message, similar_memories)
        else:
            response = await self._generate_response(user_message, similar_memories)
        
        # Store response
        self.memory.add_memory(response, "assistant")
        
        # Get memory panel
        memory_panel = self._get_memory_panel(similar_memories)
        
        return response, memory_panel
    
    async def _generate_response(self, user_message: str, similar_memories: List[Dict]) -> str:
        """Generate response using Groq."""
        try:
            # Build system prompt
            system_prompt = "You are Thread, a creative memory agent. Be helpful, creative, and concise."
            
            if similar_memories:
                system_prompt += "\n\nRelevant memories:\n"
                for mem in similar_memories:
                    text = mem.get('text_preview', mem.get('text', ''))
                    system_prompt += f"- {text}\n"
            
            # Get recent context
            recent_context = self.memory.get_recent_context(limit=5)
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            for context in recent_context:
                messages.append({
                    "role": context["role"],
                    "content": context["content"]
                })
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return self._generate_fallback_response(user_message, similar_memories)
    
    def _generate_fallback_response(self, user_message: str, similar_memories: List[Dict]) -> str:
        """Generate fallback response."""
        response = f"**🤖 Thread Agent Response**\n\n"
        response += f"📝 **Your Message:** {user_message}\n\n"
        response += "⚠️ **Status:** Creative reasoning limited (API key needed)\n\n"
        
        if similar_memories:
            response += "🧠 **Related Memories:**\n\n"
            for i, mem in enumerate(similar_memories, 1):
                text = mem.get('text_preview', '')
                similarity = mem.get('similarity', 0)
                role = mem.get('role', 'unknown')
                emoji = "👤" if role == "user" else "🤖"
                response += f"**{i}.** {emoji} {text} (relevance: {similarity:.3f})\n"
        
        response += "\n💡 **To unlock full capabilities:** Get API key from console.groq.com"
        return response
    
    def _get_memory_panel(self, similar_memories: Optional[List[Dict]] = None) -> str:
        """Generate memory panel."""
        stats = self.memory.get_stats()
        
        panel = f"### 📝 Project Memory\n"
        panel += f"**Total:** {stats['total_entries']} entries\n"
        panel += f"**Size:** {stats['index_size_bytes'] / 1024:.1f} KB\n\n"
        
        panel += "### 🔍 Similar Memories\n"
        
        if similar_memories:
            for i, mem in enumerate(similar_memories, 1):
                text = mem.get('text_preview', '')
                similarity = mem.get('similarity', 0)
                role = mem.get('role', 'unknown')
                emoji = "👤" if role == "user" else "🤖"
                panel += f"**{i}.** {emoji} {text}\n"
                panel += f"   📊 {similarity:.3f} relevance\n\n"
        else:
            panel += "*Start a conversation to build context.*\n"
        
        return panel
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        base_stats = self.memory.get_stats()
        base_stats["groq_connected"] = self.groq_client is not None
        return base_stats
    
    def clear_memory(self) -> str:
        """Clear all memories."""
        self.memory.reset()
        self.creative_intent.clear()
        return self._get_memory_panel() 