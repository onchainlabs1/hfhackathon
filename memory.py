"""
Vector-based memory system for the Thread agent using FAISS and SentenceTransformers.
Handles storage and retrieval of conversation memories through semantic similarity.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Increase timeout and add retries for model download
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

@dataclass
class MemoryEntry:
    """Represents a single memory entry with its metadata."""

    text: str
    timestamp: datetime
    source: str = "conversation"
    metadata: Dict = None

    def to_dict(self) -> Dict:
        """Convert memory entry to dictionary format."""
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata or {},
        }


class MemoryManager:
    """Vector-based memory management system using FAISS."""

    def __init__(self):
        # Try to initialize with a smaller model first
        try:
            self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            self.embedding_dim = 384
        except Exception as e:
            print(f"Failed to load primary model: {e}")
            try:
                # Fallback to an even smaller model
                self.model = SentenceTransformer("paraphrase-albert-small-v2")
                self.embedding_dim = 768
            except Exception as e:
                print(f"Failed to load fallback model: {e}")
                # Last resort: use a basic embedding
                self.model = None
                self.embedding_dim = 100

        # Initialize FAISS index for vector similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Store raw memory entries for retrieval
        self.memories: List[MemoryEntry] = []

        # Track statistics
        self.total_entries = 0

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with fallback options."""
        if self.model is not None:
            try:
                return self.model.encode([text])[0]
            except Exception as e:
                print(f"Error generating embedding with model: {e}")

        # Fallback: Generate a simple embedding based on character counts
        simple_embedding = np.zeros(self.embedding_dim)
        for i, char in enumerate(text):
            simple_embedding[i % self.embedding_dim] = ord(char)
        return simple_embedding / np.linalg.norm(simple_embedding)

    def add_entry(
        self, text: str, source: str = "conversation", metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a new memory entry to both vector index and raw storage.

        Args:
            text: The text content to store
            source: Source of the memory (default: "conversation")
            metadata: Optional metadata dictionary
        """
        # Create and store the raw memory entry
        entry = MemoryEntry(
            text=text, timestamp=datetime.now(), source=source, metadata=metadata
        )
        self.memories.append(entry)

        # Generate and add embedding to FAISS index
        try:
            embedding = self._get_embedding(text)
            embedding_normalized = embedding.reshape(1, -1).astype(np.float32)
            self.index.add(embedding_normalized)
            self.total_entries += 1
        except Exception as e:
            print(f"Error adding entry to memory: {e}")

    def retrieve_similar(
        self, query: str, top_k: int = 3, threshold: float = 0.6
    ) -> List[Dict]:
        """
        Retrieve similar memories using vector similarity.

        Args:
            query: The query text to find similar memories for
            top_k: Number of similar memories to retrieve
            threshold: Similarity threshold (0-1)

        Returns:
            List of similar memory entries with similarity scores
        """
        if self.total_entries == 0:
            return []

        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

            # Search the index
            distances, indices = self.index.search(query_embedding, min(top_k, self.total_entries))

            # Format results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.memories):  # Ensure valid index
                    similarity = 1 / (1 + dist)  # Convert distance to similarity score
                    if similarity >= threshold:
                        memory = self.memories[idx]
                        results.append({
                            **memory.to_dict(),
                            "similarity": similarity
                        })

            return sorted(results, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            print(f"Error retrieving similar memories: {e}")
            return []

    def reset(self) -> None:
        """Reset the memory system."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.memories.clear()
        self.total_entries = 0

    def get_stats(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dictionary containing memory statistics
        """
        latest_timestamp = None
        if self.memories:
            latest_timestamp = max(m.timestamp for m in self.memories)

        return {
            "total_entries": self.total_entries,
            "index_size_bytes": self.total_entries * self.embedding_dim * 4,  # 4 bytes per float
            "embedding_dim": self.embedding_dim,
            "latest_timestamp": latest_timestamp,
        }

    def get_all_entries(self) -> List[Dict]:
        """Get all memory entries in chronological order."""
        return [entry.to_dict() for entry in self.memories]
