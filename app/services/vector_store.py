"""
vector_store.py

This module handles storage and retrieval of text embeddings using ChromaDB and OpenAI.
It supports storing document chunks along with metadata and querying the most semantically similar content.
"""

import os
import logging
import chromadb
from groq import Groq
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app.core.config import settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Get GROQ API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Configure logging
logger = logging.getLogger(__name__)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def initialize_vector_store():
    """Initialize ChromaDB client with error handling"""
    try:
        # Ensure directory exists
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
        
        # Initialize ChromaDB
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        collection = client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Successfully initialized ChromaDB at {settings.CHROMA_DB_PATH}")
        
        return client, collection
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

# Initialize components
client, collection = initialize_vector_store()

Embedding = List[float]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using Sentence Transformers with normalization"""
    try:
        if not texts:
            return []
            
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            
        # Clean and validate inputs
        validated_texts = [str(text).strip() for text in texts if str(text).strip()]
        if not validated_texts:
            logger.warning("No valid text to embed after cleaning")
            return []
            
        # Use sentence transformers with normalization based on settings
        embeddings = embedder.encode(
            validated_texts, 
            convert_to_tensor=False, 
            normalize_embeddings=getattr(settings, 'NORMALIZE_EMBEDDINGS', True)
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

def batch_encode(texts: List[str]) -> List[List[float]]:
    """
    Encode a batch of texts using OpenAI embeddings API
    
    Args:
        texts (List[str]): List of text strings to encode
        
    Returns:
        List[List[float]]: List of embeddings
    """
    try:
        return get_embeddings(texts)
    except Exception as e:
        logger.error(f"Error encoding batch: {str(e)}")
        raise

def store_text_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stores text chunks in ChromaDB along with their sentence embeddings.

    Args:
        chunks (List[Dict]): List of dictionaries containing:
            - 'content' (str): Text content of the chunk
            - 'meta' (Dict): Metadata dictionary with at least 'source' (filename/origin)

    Returns:
        Dict: Status of the storage operation
    """
    try:
        if not chunks:
            return {"status": "error", "message": "No chunks provided"}

        # Get current collection size for ID generation
        current_size = len(collection.get()['ids'])
        
        # Prepare batch data
        contents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['meta'] for chunk in chunks]
        ids = [f"chunk_{current_size + i}" for i in range(len(chunks))]

        # Process in batches
        total_chunks = len(chunks)
        processed = 0
        batch_size = settings.BATCH_SIZE

        while processed < total_chunks:
            batch_end = min(processed + batch_size, total_chunks)
            batch_slice = slice(processed, batch_end)
            
            batch_contents = contents[batch_slice]
            batch_metadatas = metadatas[batch_slice]
            batch_ids = ids[batch_slice]

            # Generate embeddings for the batch using OpenAI
            try:
                batch_embeddings = get_embeddings(batch_contents)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                return {"status": "error", "message": f"OpenAI embedding generation failed: {str(e)}"}

            # Add to ChromaDB
            try:
                # Convert embeddings to numpy array for ChromaDB (fix deprecated np.float64)
                embeddings_array = np.array([np.array(emb, dtype=np.float32) for emb in batch_embeddings])
                collection.add(
                    embeddings=embeddings_array.tolist(),  # ChromaDB expects List[List[float]]
                    documents=batch_contents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                processed += len(batch_contents)
                logger.info(f"Processed {processed}/{total_chunks} chunks")
                 
            
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {str(e)}")
                return {"status": "error", "message": f"ChromaDB storage failed: {str(e)}"}

        return {
            "status": "success",
            "message": f"Successfully stored {total_chunks} chunks",
            "chunks_stored": total_chunks
            
        }
        print("total_chunks", total_chunks)
        print("processed", processed, "unprocessed", total_chunks - processed)

    except Exception as e:
        logger.error(f"Error in store_text_chunks: {str(e)}")
        return {"status": "error", "message": str(e)}

# Alias for backward compatibility
def search_similar(query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
    return query_similar_chunks(query_text, n_results)

def query_similar_chunks(query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """
    Query the vector store for chunks similar to the input text.
    
    Args:
        query_text (str): The text to find similar chunks for
        n_results (int): Number of results to return
        
    Returns:
        List[Dict]: List of similar chunks with their metadata and similarity scores
    """
    try:
        # Generate query embedding using OpenAI
        query_embedding = get_embeddings([query_text])[0]
        
        # Query the collection
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            return []

        # Format results
        formatted_results = []
        try:
            if results and isinstance(results, dict):
                # Safely get results
                ids_list = results.get('ids', [])
                docs_list = results.get('documents', [])
                meta_list = results.get('metadatas', [])
                dist_list = results.get('distances', [])

                if ids_list and len(ids_list) > 0:
                    ids = ids_list[0]
                    documents = docs_list[0] if docs_list else []
                    metadatas = meta_list[0] if meta_list else []
                    distances = dist_list[0] if dist_list else []

                    for i in range(len(ids)):
                        formatted_results.append({
                            "content": documents[i] if i < len(documents) else "",
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "similarity": 1 - float(distances[i]) if i < len(distances) else 0.0
                        })
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            return []
            
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        return []

def get_stored_documents() -> Dict[str, Any]:
    """Get information about documents currently stored in the vector database"""
    try:
        # Get all documents from the collection
        all_data = collection.get(include=["documents", "metadatas"])
        
        if not all_data or not all_data.get('metadatas'):
            return {"total_chunks": 0, "documents": {}, "sources": []}
        
        metadatas = all_data.get('metadatas', [])
        if not metadatas:
            return {"total_chunks": 0, "documents": {}, "sources": []}
        
        # Count documents by source
        doc_counts = {}
        for metadata in metadatas:
            source = metadata.get('source', 'unknown') if metadata else 'unknown'
            doc_counts[source] = doc_counts.get(source, 0) + 1
        
        return {
            "total_chunks": len(metadatas),
            "documents": doc_counts,
            "sources": list(doc_counts.keys())
        }
    except Exception as e:
        logger.error(f"Error getting stored documents: {str(e)}")
        return {"error": str(e)}

def clear_vector_store():
    """Clear all documents from the vector store"""
    try:
        global collection
        # Delete the collection
        client.delete_collection("docs")
        # Recreate it
        collection = client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store cleared successfully")
        return {"status": "success", "message": "Vector store cleared"}
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        return {"status": "error", "message": str(e)}

def search_similar_improved(query: str, source: Optional[str] = None, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
    """Search for similar chunks with proper relevance ranking"""
    try:
        # Set default value for n_results
        default_count = getattr(settings, 'TOP_K_RETRIEVED_CHUNKS', 10)
        actual_count = n_results if n_results is not None else default_count
        actual_count = max(1, actual_count)
        
        # Get results from vector store (get more for better filtering)
        search_count = actual_count * 2
        results = query_similar_chunks(query, search_count)
        
        if not results:
            logger.warning("No results returned from vector store")
            return []
        
        # Filter by source if specified
        if source:
            results = [
                r for r in results 
                if r.get("metadata", {}).get("source", "").lower() == source.lower()
            ]
        
        # Sort by similarity score (highest first)
        results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Apply minimum similarity threshold for relevance
        min_similarity = 0.1  # Minimum threshold for relevance
        relevant_results = [r for r in results if r.get("similarity", 0) >= min_similarity]
        
        if not relevant_results:
            # If no results meet threshold, take top 3 results anyway
            relevant_results = results[:3]
            logger.warning(f"No results above similarity threshold {min_similarity}, using top 3")
        
        # Limit to requested number of results
        final_results = relevant_results[:actual_count]
        
        # Format results for synthesis
        formatted_results = []
        for result in final_results:
            formatted_result = {
                "content": result.get("content", ""),
                "meta": {
                    "source": result.get("metadata", {}).get("source", "unknown"),
                    "similarity": result.get("similarity", 0),
                    "chunk_index": result.get("metadata", {}).get("chunk_index", 0),
                    "total_chunks": result.get("metadata", {}).get("total_chunks", 1)
                }
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"Returning {len(formatted_results)} relevant results for query: {query[:50]}...")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in search_similar_improved: {str(e)}")
        return []
