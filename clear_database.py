"""
Script to clear all documents from the ChromaDB database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import clear_vector_store
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Clear all documents from the vector database"""
    try:
        print("Clearing all documents from ChromaDB...")
        result = clear_vector_store()
        print(f"Result: {result}")
        
        if result.get("status") == "success":
            print("✅ Database cleared successfully!")
            print("You can now re-upload your documents with the improved chunking (overlap=250)")
        else:
            print("❌ Failed to clear database")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
