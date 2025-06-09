"""
🧠 SEMANTIC MEMORY MODULE
Vector-based memory system using SentenceTransformers + FAISS
Aligned with Model Context Protocol (MCP) specifications
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class MemoryEntry:
    """Data structure for memory entries with metadata."""
    
    def __init__(
        self,
        text: str,
        role: str,  # "user" or "assistant"
        timestamp: datetime,
        metadata: Optional[Dict] = None
    ):
        self.text = text
        self.role = role
        self.timestamp = timestamp
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for retrieval."""
        return {
            "text": self.text,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class SemanticMemory:
    """
    Semantic memory system using SentenceTransformers + FAISS.
    Stores and retrieves memories based on semantic similarity.
    """
    
    def __init__(self):
        # Initialize SentenceTransformer model
        print("🧠 Initializing semantic memory...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store memory entries with metadata
        self.memories: List[MemoryEntry] = []
        self.total_entries = 0
    
    def add_memory(self, text: str, role: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a new memory entry with semantic embedding.
        
        Args:
            text: The message content
            role: "user" or "assistant"
            metadata: Optional additional metadata
        """
        if not text.strip():
            return
            
        # Create memory entry
        entry = MemoryEntry(
            text=text,
            role=role,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Generate embedding
        embedding = self.model.encode([text])[0]
        embedding_normalized = embedding.reshape(1, -1).astype(np.float32)
        
        # Add to FAISS index
        self.index.add(embedding_normalized)
        
        # Store memory entry
        self.memories.append(entry)
        self.total_entries += 1
        
        print(f"💾 Memory added: {role} - {text[:50]}...")
    
    def retrieve_similar(
        self, 
        message: str, 
        top_k: int = 3, 
        exclude_current: bool = True
    ) -> List[Dict]:
        """
        Retrieve semantically similar memories.
        
        Args:
            message: Query message
            top_k: Number of similar memories to return
            exclude_current: Whether to exclude the current message
            
        Returns:
            List of similar memories with similarity scores
        """
        if self.total_entries == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([message])[0]
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        search_k = min(top_k + (1 if exclude_current else 0), self.total_entries)
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.memories):
                memory = self.memories[idx]
                
                # Skip if excluding current message (exact match)
                if exclude_current and memory.text.strip() == message.strip():
                    continue
                
                # Calculate similarity (0-1 scale)
                similarity = 1.0 / (1.0 + dist)
                
                # Format memory entry
                memory_dict = memory.to_dict()
                memory_dict["similarity"] = round(similarity, 3)
                memory_dict["text_preview"] = self._truncate_text(memory.text, 80)
                memory_dict["formatted_timestamp"] = memory.timestamp.strftime("%m/%d %H:%M")
                
                results.append(memory_dict)
                
                if len(results) >= top_k:
                    break
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    def get_recent_context(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation context."""
        recent_memories = self.memories[-limit:] if self.memories else []
        return [
            {
                "role": mem.role,
                "content": mem.text,
                "timestamp": mem.timestamp.isoformat()
            }
            for mem in recent_memories
        ]
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        latest_timestamp = None
        if self.memories:
            latest_timestamp = self.memories[-1].timestamp
        
        return {
            "total_entries": self.total_entries,
            "index_size_bytes": self.total_entries * self.embedding_dim * 4,
            "embedding_dim": self.embedding_dim,
            "latest_timestamp": latest_timestamp
        }
    
    def reset(self) -> None:
        """Reset all memories."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.memories.clear()
        self.total_entries = 0
        print("🗑️ Memory reset complete")


# MCP-aligned memory endpoints (placeholder for future MCP integration)
class MCPMemoryServer:
    """Placeholder MCP server endpoints for memory operations."""
    
    def __init__(self, memory: SemanticMemory):
        self.memory = memory
    
    async def context_endpoint(self, request: Dict) -> Dict:
        """MCP /context endpoint"""
        return {
            "context": self.memory.get_recent_context(),
            "total_memories": self.memory.total_entries
        }
    
    async def memory_add_endpoint(self, request: Dict) -> Dict:
        """MCP /memory/add endpoint"""
        text = request.get("text", "")
        role = request.get("role", "user")
        metadata = request.get("metadata", {})
        
        self.memory.add_memory(text, role, metadata)
        return {"success": True, "message": "Memory added"}
    
    async def memory_search_endpoint(self, request: Dict) -> Dict:
        """MCP /memory/search endpoint"""
        query = request.get("query", "")
        top_k = request.get("top_k", 3)
        
        results = self.memory.retrieve_similar(query, top_k)
        return {"results": results}
