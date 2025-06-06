"""Simple text-based memory management system."""

import re
from datetime import datetime
from typing import Dict, List, Optional


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
    """Simple text-based memory management system."""

    def __init__(self):
        # Store raw memory entries
        self.memories: List[MemoryEntry] = []
        self.total_entries = 0

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for similarity matching."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter out short words
        words = [word for word in text.split() if len(word) > 2]
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 
            'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'they', 'have', 'from', 'been', 'said', 
            'each', 'which', 'would', 'there', 'their', 'what', 'make', 'about', 'time', 'very', 'when', 'come', 
            'could', 'like', 'into', 'him', 'than', 'find', 'call', 'down', 'first', 'look', 'made', 'more', 'most', 
            'move', 'much', 'must', 'name', 'need', 'number', 'other', 'over', 'part', 'place', 'right', 'same', 
            'seem', 'show', 'side', 'tell', 'turn', 'want', 'well', 'were'
        }
        
        return [word for word in words if word not in stop_words]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using keyword overlap."""
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0

    def add_entry(
        self, text: str, source: str = "conversation", metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a new memory entry.

        Args:
            text: The text content to store
            source: Source of the memory (default: "conversation")
            metadata: Optional metadata dictionary
        """
        try:
            # Create and store the memory entry
            entry = MemoryEntry(
                text=text, timestamp=datetime.now(), source=source, metadata=metadata
            )
            self.memories.append(entry)
            self.total_entries += 1
        except Exception as e:
            print(f"Error adding entry to memory: {e}")

    def retrieve_similar(
        self, query: str, top_k: int = 3, threshold: float = 0.1
    ) -> List[Dict]:
        """
        Retrieve similar memories using text similarity.

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
            # Calculate similarity scores for all memories
            similarities = []
            for memory in self.memories:
                similarity = self._calculate_similarity(query, memory.text)
                if similarity >= threshold:
                    similarities.append({
                        **memory.to_dict(),
                        "similarity": round(similarity, 3)
                    })

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            print(f"Error retrieving similar memories: {e}")
            return []

    def reset(self) -> None:
        """Reset the memory system."""
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

        # Calculate approximate size
        total_text_size = sum(len(memory.text) for memory in self.memories)

        return {
            "total_entries": self.total_entries,
            "index_size_bytes": total_text_size,  # Approximate size in bytes
            "embedding_dim": 0,  # No embeddings used
            "latest_timestamp": latest_timestamp,
        }

    def get_all_entries(self) -> List[Dict]:
        """Get all memory entries in chronological order."""
        return [entry.to_dict() for entry in self.memories]
