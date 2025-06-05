"""
Unit tests for the Thread memory system.
Tests the core functionality of MemoryManager including adding entries and retrieving similar memories.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import MemoryManager


class TestMemoryManager:
    """Test class for MemoryManager functionality."""

    def test_add_entry_basic(self):
        """Test basic functionality of add_entry method."""
        memory = MemoryManager()

        # Test adding a simple entry
        text = "Python is a programming language"
        memory.add_entry(text, source="test")

        # Verify entry was added
        assert memory.total_entries == 1
        assert len(memory.memories) == 1

        # Verify entry content
        stored_entry = memory.memories[0]
        assert stored_entry.text == text
        assert stored_entry.source == "test"
        assert isinstance(stored_entry.timestamp, datetime)

        print("✅ test_add_entry_basic passed")

    def test_add_entry_multiple(self):
        """Test adding multiple entries."""
        memory = MemoryManager()

        test_entries = [
            ("First entry about Python programming", "user"),
            ("Second entry about machine learning", "assistant"),
            ("Third entry about data science", "user"),
        ]

        # Add multiple entries
        for text, source in test_entries:
            memory.add_entry(text, source=source)

        # Verify all entries were added
        assert memory.total_entries == 3
        assert len(memory.memories) == 3

        # Verify entries are in correct order
        for i, (expected_text, expected_source) in enumerate(test_entries):
            assert memory.memories[i].text == expected_text
            assert memory.memories[i].source == expected_source

        print("✅ test_add_entry_multiple passed")

    def test_retrieve_similar_empty(self):
        """Test retrieve_similar with empty memory."""
        memory = MemoryManager()

        # Test retrieval from empty memory
        results = memory.retrieve_similar("test query", top_k=3)

        assert results == []
        print("✅ test_retrieve_similar_empty passed")

    def test_retrieve_similar_basic(self):
        """Test basic functionality of retrieve_similar method."""
        memory = MemoryManager()

        # Add test entries
        test_entries = [
            "Python is a programming language used for development",
            "Machine learning models require training data",
            "Data science involves analyzing datasets",
            "Web development with Python Django framework",
            "JavaScript is used for frontend development",
        ]

        for text in test_entries:
            memory.add_entry(text, source="test")

        # Test similarity search for programming-related query
        query = "programming languages and development"
        results = memory.retrieve_similar(query, top_k=3)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all("text" in result for result in results)
        assert all("similarity" in result for result in results)
        assert all("timestamp" in result for result in results)
        assert all("source" in result for result in results)

        # Verify similarity scores are reasonable
        for result in results:
            assert 0.0 <= result["similarity"] <= 1.0

        # Verify results are sorted by similarity (highest first)
        similarities = [result["similarity"] for result in results]
        assert similarities == sorted(similarities, reverse=True)

        print("✅ test_retrieve_similar_basic passed")

    def test_retrieve_similar_top_k_limit(self):
        """Test that retrieve_similar respects top_k parameter."""
        memory = MemoryManager()

        # Add 5 test entries
        for i in range(5):
            memory.add_entry(f"Test entry number {i}", source="test")

        # Test different top_k values
        for k in [1, 2, 3, 5, 10]:
            results = memory.retrieve_similar("test query", top_k=k)
            expected_length = min(k, 5)  # Should not exceed total entries
            assert len(results) == expected_length

        print("✅ test_retrieve_similar_top_k_limit passed")

    def test_memory_stats(self):
        """Test get_stats method."""
        memory = MemoryManager()

        # Test stats with empty memory
        stats = memory.get_stats()
        assert stats["total_entries"] == 0
        assert stats["latest_timestamp"] is None

        # Add entries and test stats
        memory.add_entry("Test entry 1", source="test")
        memory.add_entry("Test entry 2", source="test")

        stats = memory.get_stats()
        assert stats["total_entries"] == 2
        assert stats["embedding_dim"] == 384  # all-MiniLM-L6-v2 dimension
        assert stats["latest_timestamp"] is not None
        assert isinstance(stats["latest_timestamp"], datetime)

        print("✅ test_memory_stats passed")

    def test_memory_reset(self):
        """Test reset functionality."""
        memory = MemoryManager()

        # Add some entries
        memory.add_entry("Test entry 1", source="test")
        memory.add_entry("Test entry 2", source="test")

        # Verify entries exist
        assert memory.total_entries == 2
        assert len(memory.memories) == 2

        # Reset memory
        memory.reset()

        # Verify memory is cleared
        assert memory.total_entries == 0
        assert len(memory.memories) == 0

        # Verify we can still add entries after reset
        memory.add_entry("New entry after reset", source="test")
        assert memory.total_entries == 1

        print("✅ test_memory_reset passed")


def run_all_tests():
    """Run all memory tests."""
    print("🧪 Running Thread Memory System Tests")
    print("-" * 50)

    test_instance = TestMemoryManager()

    try:
        test_instance.test_add_entry_basic()
        test_instance.test_add_entry_multiple()
        test_instance.test_retrieve_similar_empty()
        test_instance.test_retrieve_similar_basic()
        test_instance.test_retrieve_similar_top_k_limit()
        test_instance.test_memory_stats()
        test_instance.test_memory_reset()

        print("-" * 50)
        print("✨ All tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
