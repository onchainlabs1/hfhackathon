"""
🔧 HF SPACES GROQ FIX - Minimal test for deployment
Run this to verify Groq initialization works correctly
"""

import os
import sys

def test_groq_initialization():
    """Test Groq client initialization with multiple methods."""
    
    # Set test API key
    os.environ['GROQ_API_KEY'] = 'gsk_test_key_12345'
    
    print("🧪 Testing Groq initialization methods...")
    
    # Method 1: Clear cache and import fresh
    if 'groq' in sys.modules:
        del sys.modules['groq']
        print("✅ Cleared groq from sys.modules")
    
    try:
        import groq
        print(f"📦 Groq version: {groq.__version__}")
        
        # Try to create client
        client = groq.Groq(api_key=os.environ['GROQ_API_KEY'])
        print("✅ Groq client created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Primary method failed: {e}")
        
        # Fallback method
        try:
            import importlib
            groq_module = importlib.import_module('groq')
            GroqClass = getattr(groq_module, 'Groq')
            client = GroqClass(api_key=os.environ['GROQ_API_KEY'])
            print("✅ Groq client created with fallback!")
            return True
            
        except Exception as final_e:
            print(f"❌ All methods failed: {final_e}")
            return False

if __name__ == "__main__":
    success = test_groq_initialization()
    if success:
        print("🎉 GROQ INITIALIZATION WORKS!")
    else:
        print("💥 GROQ INITIALIZATION FAILED!") 