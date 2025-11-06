from fastapi import APIRouter
from typing import Optional
from ...services.vector_store import collection, query_similar_chunks, clear_vector_store
import logging

logger = logging.getLogger(__name__)
debug_router = APIRouter()

@debug_router.get("/database-stats")
def get_database_stats():
    """Get current database statistics"""
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data:
            return {"total_documents": 0, "sources": {}}
            
        documents = all_data.get('documents') or []
        metadatas = all_data.get('metadatas') or []
        
        # Count by source
        sources = {}
        if metadatas:
            for meta in metadatas:
                if meta and isinstance(meta, dict):
                    source = meta.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
            
        return {
            "total_documents": len(documents),
            "sources": sources,
            "collection_name": collection.name
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {"error": str(e)}

@debug_router.get("/documents")
def get_document_overview():
    """Get overview of stored documents with sample content"""
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data:
            return {"total_documents": 0, "sources": {}, "samples": []}
            
        documents = all_data.get('documents') or []
        metadatas = all_data.get('metadatas') or []
        
        if not documents:
            return {"total_documents": 0, "sources": {}, "samples": []}
        
        # Count by source and collect samples
        sources = {}
        samples = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            if meta and isinstance(meta, dict):
                source = meta.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                
                # Add sample content from first few documents
                if len(samples) < 5:
                    samples.append({
                        'source': source,
                        'content_preview': doc[:200] + '...' if len(doc) > 200 else doc,
                        'content_length': len(doc)
                    })
        
        return {
            "total_documents": len(documents),
            "sources": sources,
            "samples": samples
        }
    except Exception as e:
        logger.error(f"Error getting document overview: {str(e)}")
        return {"error": str(e)}

@debug_router.get("/search/{query}")
def test_search(query: str):
    """Test search functionality with detailed results"""
    try:
        logger.info(f"Testing search for: '{query}'")
        
        # Get results from vector store
        results = query_similar_chunks(query, 10)
        
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "similarity": result.get("similarity", 0),
                "source": result.get("metadata", {}).get("source", "unknown"),
                "content_preview": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "content_length": len(result.get("content", ""))
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"Error in test search: {str(e)}")
        return {"error": str(e)}

@debug_router.get("/content")
def search_content(query: str = "example", source: Optional[str] = None):
    """Search for content containing specific terms"""
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data:
            return {"query_term": query, "total_chunks": 0, "matching_chunks": []}
            
        documents = all_data.get('documents') or []
        metadatas = all_data.get('metadatas') or []
        
        matching_chunks = []
        for doc, meta in zip(documents, metadatas):
            if meta and isinstance(meta, dict):
                doc_source = meta.get('source', 'unknown')
                
                # Filter by source if specified
                if source and source.lower() not in str(doc_source).lower():
                    continue
                    
                # Check if query term is in content
                contains_query = query.lower() in doc.lower()
                matching_chunks.append({
                    'source': doc_source,
                    'content_preview': doc[:300] + '...' if len(doc) > 300 else doc,
                    'contains_query': contains_query,
                    'content_length': len(doc)
                })
        
        return {
            "query_term": query,
            "source_filter": source,
            "total_chunks": len(matching_chunks),
            "chunks_containing_query": sum(1 for c in matching_chunks if c['contains_query']),
            "matching_chunks": matching_chunks[:5]  # Show first 5
        }
    except Exception as e:
        logger.error(f"Error searching content: {str(e)}")
        return {"error": str(e)}

@debug_router.post("/clear-database")
def clear_database():
    """Clear all documents and embeddings from ChromaDB"""
    try:
        result = clear_vector_store()
        logger.info("Database cleared successfully")
        return {
            "status": "success", 
            "message": "All documents and embeddings cleared", 
            "result": result
        }
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return {"status": "error", "message": str(e)}
