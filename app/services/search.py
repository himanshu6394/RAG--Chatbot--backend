"""
search.py

This module handles semantic search functionality using the vector store.
"""

import logging
from typing import List, Dict, Any, Optional
from .vector_store import query_similar_chunks, search_similar_improved

logger = logging.getLogger(__name__)

def search_similar(query: str, source: Optional[str] = None, n_results: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced search with detailed logging to debug resume content issues
    """
    try:
        logger.info(f"=== SEARCH DEBUG === Query: '{query}', Source: {source}, n_results: {n_results}")
        
        # Get results from vector store with extra results for filtering
        raw_results = search_similar_improved(query, source=source, n_results=n_results * 2)
        
        logger.info(f"Raw results from vector store: {len(raw_results)}")
        
        if not raw_results:
            logger.warning("No results from vector store - this suggests the content isn't stored properly")
            return []
        
        # Log what we actually got for debugging
        for i, result in enumerate(raw_results[:5]):  # Show first 5 results
            similarity = result.get("meta", {}).get("similarity", 0)
            source_name = result.get("meta", {}).get("source", "unknown")
            content = result.get("content", "")
            content_preview = content[:150].replace('\n', ' ') + "..." if len(content) > 150 else content
            logger.info(f"Result {i+1}: similarity={similarity:.4f}, source='{source_name}', content='{content_preview}'")
        
        # For resume-related queries, use very low threshold
        query_lower = query.lower()
        resume_keywords = ['skill', 'experience', 'project', 'education', 'resume', 'cv', 'qualification']
        is_resume_query = any(keyword in query_lower for keyword in resume_keywords)
        
        if is_resume_query:
            # Use very low threshold for resume queries
            min_similarity = 0.01
            logger.info(f"Resume query detected - using very low threshold: {min_similarity}")
        else:
            min_similarity = 0.1
        
        # Filter by similarity
        filtered_results = [r for r in raw_results if r.get("meta", {}).get("similarity", 0) >= min_similarity]
        logger.info(f"After similarity filtering (>={min_similarity}): {len(filtered_results)} results")
        
        # If we have specific source, filter by it
        if source:
            source_filtered = [r for r in filtered_results if source.lower() in r.get("meta", {}).get("source", "").lower()]
            logger.info(f"After source filtering for '{source}': {len(source_filtered)} results")
            filtered_results = source_filtered
        
        # Sort by similarity (highest first)
        filtered_results.sort(key=lambda x: x.get("meta", {}).get("similarity", 0), reverse=True)
        
        # If still no results for resume query, be more aggressive
        if not filtered_results and is_resume_query:
            logger.warning("No results for resume query - trying with all results")
            filtered_results = raw_results[:5]  # Take top 5 regardless of similarity
        
        # Take final results
        final_results = filtered_results[:n_results]
        
        # Final logging
        similarities = [r.get("meta", {}).get("similarity", 0) for r in final_results]
        sources = [r.get("meta", {}).get("source", "unknown") for r in final_results]
        logger.info(f"=== FINAL RESULTS === Count: {len(final_results)}, Similarities: {similarities}, Sources: {sources}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in search_similar: {str(e)}")
        return []
