"""
Vector-based memory system for the Thread agent using FAISS and SentenceTransformers.
Handles storage and retrieval of conversation memories through semantic similarity.
"""

from datetime import datetime
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedding_model():
    """Get the embedding model with persistent cache for Hugging Face Spaces."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="/data/models")


class MemoryEntry:
    """Data class for storing memory entries."""

    def __init__(
        self,
        text: str,
        timestamp: datetime,
        source: str = "conversation",
        metadata: Optional[Dict] = None,
    ):
        self.text = text
        self.timestamp = timestamp
        self.source = source
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert entry to dictionary format."""
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


class MemoryManager:
    """Vector-based memory management system using FAISS."""

    def __init__(self):
        # Initialize the sentence transformer model with optimized caching
        self.model = get_embedding_model()
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2 embeddings

        # Initialize FAISS index for vector similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Store raw memory entries for retrieval
        self.memories: List[MemoryEntry] = []

        # Track statistics
        self.total_entries = 0

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
        embedding = self.model.encode([text])[0]  # Get the first (and only) embedding
        embedding_normalized = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding_normalized)

        self.total_entries += 1

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the top-k most similar memories to the query.

        Args:
            query: The text to find similar memories for
            top_k: Number of similar memories to retrieve

        Returns:
            List of memory entries with similarity scores
        """
        if self.total_entries == 0:
            return []

        # Adjust top_k if we have fewer entries
        top_k = min(top_k, self.total_entries)

        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            memory = self.memories[idx]
            similarity_score = 1.0 / (
                1.0 + distance
            )  # Convert distance to similarity score

            results.append(
                {**memory.to_dict(), "similarity": round(similarity_score, 3)}
            )

        return results

    def reset(self) -> None:
        """Clear all memories and reset the FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.memories.clear()
        self.total_entries = 0

    def get_stats(self) -> Dict:
        """Get current memory statistics."""
        return {
            "total_entries": self.total_entries,
            "index_size_bytes": self.index.ntotal
            * self.embedding_dim
            * 4,  # 4 bytes per float32
            "embedding_dim": self.embedding_dim,
            "latest_timestamp": self.memories[-1].timestamp if self.memories else None,
        }

    def get_all_entries(self) -> List[Dict]:
        """Get all memory entries in chronological order."""
        return [entry.to_dict() for entry in self.memories]
