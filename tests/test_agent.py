"""
Unit tests for the Thread agent system.
Tests the core functionality of ThreadAgent including initialization, topic detection, and prompt building.
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ThreadAgent


class TestThreadAgent:
    """Test class for ThreadAgent functionality."""

    def test_agent_initialization(self):
        """Test basic ThreadAgent initialization."""
        agent = ThreadAgent()

        # Verify basic attributes
        assert agent.current_topic == "General Discussion"
        assert agent.suggested_next_step == "Start a conversation to explore topics"
        assert agent.conversation_history == []
        assert agent.memory is not None

        # Verify topic keywords are properly loaded
        assert "pricing" in agent.topic_keywords
        assert "technical" in agent.topic_keywords
        assert len(agent.topic_keywords) == 7

        # Verify system prompt is set
        assert "Thread" in agent.base_system_prompt
        assert "memory management" in agent.base_system_prompt

        print("✅ test_agent_initialization passed")

    def test_topic_detection(self):
        """Test topic detection functionality."""
        agent = ThreadAgent()

        # Test technical topic detection
        tech_message = "We have a bug in our Python code that needs fixing"
        detected_topic = agent._detect_topic(tech_message)
        assert detected_topic == "Technical"

        # Test pricing topic detection
        pricing_message = "What's the cost of this service and pricing options?"
        detected_topic = agent._detect_topic(pricing_message)
        assert detected_topic == "Pricing"

        # Test business topic detection
        business_message = "Our sales strategy needs improvement for market growth"
        detected_topic = agent._detect_topic(business_message)
        assert detected_topic == "Business"

        # Test fallback to current topic
        generic_message = "Hello there, how are you?"
        detected_topic = agent._detect_topic(generic_message)
        assert detected_topic == "General Discussion"  # Should keep current topic

        print("✅ test_topic_detection passed")

    def test_next_step_suggestions(self):
        """Test next step suggestion generation."""
        agent = ThreadAgent()

        # Test technical suggestions
        tech_suggestion = agent._generate_next_step_suggestion(
            "We have a bug in the system", "Technical"
        )
        assert "troubleshooting" in tech_suggestion.lower()

        # Test pricing suggestions
        pricing_suggestion = agent._generate_next_step_suggestion(
            "Compare pricing plans", "Pricing"
        )
        assert "pricing" in pricing_suggestion.lower()

        # Test question-based suggestions
        question_suggestion = agent._generate_next_step_suggestion(
            "How do we solve this problem?", "General"
        )
        assert "research" in question_suggestion.lower() or "expert" in question_suggestion.lower()

        print("✅ test_next_step_suggestions passed")

    def test_enhanced_system_prompt_building(self):
        """Test enhanced system prompt construction."""
        agent = ThreadAgent()

        memory_context = "Previous conversation about Python programming"
        topic = "Technical"

        prompt = agent._build_enhanced_system_prompt(memory_context, topic)

        # Verify prompt contains expected elements
        assert agent.base_system_prompt in prompt
        assert topic in prompt
        assert memory_context in prompt
        assert "CURRENT CONTEXT" in prompt
        assert "INSTRUCTIONS" in prompt

        print("✅ test_enhanced_system_prompt_building passed")

    def test_memory_context_formatting(self):
        """Test memory context formatting."""
        agent = ThreadAgent()

        # Test with empty memories
        empty_context = agent._format_memory_context([])
        assert "No relevant memories found" in empty_context

        # Test with sample memories
        sample_memories = [
            {
                "text": "Python is a programming language",
                "source": "user",
                "similarity": 0.85
            },
            {
                "text": "Machine learning is exciting",
                "source": "assistant",
                "similarity": 0.72
            }
        ]

        formatted_context = agent._format_memory_context(sample_memories)
        assert "Relevant memories:" in formatted_context
        assert "Python is a programming language" in formatted_context
        assert "0.85" in formatted_context
        assert "[user]" in formatted_context

        print("✅ test_memory_context_formatting passed")

    def test_groq_client_reload(self):
        """Test Groq client reload functionality."""
        agent = ThreadAgent()

        # Test reload with no API key
        with patch.dict(os.environ, {}, clear=True):
            success = agent.reload_groq_client()
            assert success is False
            assert agent.groq_client is None

        # Test reload with valid API key
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=True):
            with patch("agent.Groq") as mock_groq:
                mock_groq.return_value = MagicMock()
                success = agent.reload_groq_client()
                assert success is True
                assert agent.groq_client is not None

        print("✅ test_groq_client_reload passed")

    def test_memory_panel_generation(self):
        """Test memory panel content generation."""
        agent = ThreadAgent()

        # Add some test data to memory
        agent.memory.add_entry("Test entry 1", source="user")
        agent.memory.add_entry("Test entry 2", source="assistant")

        panel_content = agent._get_memory_panel()

        # Verify panel contains expected information
        assert "Memory Statistics" in panel_content
        assert "Total Entries: 2" in panel_content
        assert "Embedding Dimension" in panel_content
        assert "Vector Index Size" in panel_content

        print("✅ test_memory_panel_generation passed")

    def test_memory_clearing(self):
        """Test memory clearing functionality."""
        agent = ThreadAgent()

        # Add some test data
        agent.memory.add_entry("Test entry", source="user")
        agent.conversation_history.append({"role": "user", "content": "test"})

        # Verify data exists
        assert len(agent.memory.memories) == 1
        assert len(agent.conversation_history) == 1

        # Clear memory
        panel_content = agent.clear_memory()

        # Verify everything is cleared
        assert len(agent.memory.memories) == 0
        assert len(agent.conversation_history) == 0
        assert "Total Entries: 0" in panel_content

        print("✅ test_memory_clearing passed")

    def test_api_configuration_check(self):
        """Test API configuration checking."""
        agent = ThreadAgent()

        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            agent.groq_client = None
            assert agent.is_api_configured() is False

        # Test with configured client
        agent.groq_client = MagicMock()
        assert agent.is_api_configured() is True

        print("✅ test_api_configuration_check passed")


def run_all_tests():
    """Run all agent tests."""
    print("🧪 Running Thread Agent System Tests")
    print("-" * 50)

    test_instance = TestThreadAgent()

    try:
        test_instance.test_agent_initialization()
        test_instance.test_topic_detection()
        test_instance.test_next_step_suggestions()
        test_instance.test_enhanced_system_prompt_building()
        test_instance.test_memory_context_formatting()
        test_instance.test_groq_client_reload()
        test_instance.test_memory_panel_generation()
        test_instance.test_memory_clearing()
        test_instance.test_api_configuration_check()

        print("-" * 50)
        print("✨ All agent tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 