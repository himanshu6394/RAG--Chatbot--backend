from chromadb import PersistentClient

# Connect to your Chroma DB using absolute path
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
client = PersistentClient(path=CHROMA_PATH)

# List collections
collections = client.list_collections()
print("Available Collections:", collections)

# If you know your collection name (often "documents" or similar)
for col in collections:
    collection = client.get_collection(col.name)
    print(f"\nCollection: {col.name}")
    print("Total docs:", collection.count())

    # Fetch a few documents
    docs = collection.get(include=["metadatas", "documents"], limit=5)
    if docs["documents"] is not None:
        for i, doc in enumerate(docs["documents"]):
            print(f"Doc {i+1}: {doc}")
            if docs["metadatas"] is not None:
                print("Metadata:", docs["metadatas"][i])
            else:
                print("Metadata: None")
    else:
        print("No documents found in this collection.")
