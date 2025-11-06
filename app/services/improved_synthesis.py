"""
Improved synthesis service with better similarity thresholds
"""
import logging
from typing import List, Dict, Any, Optional
from .synthesis import summarize_themes

logger = logging.getLogger(__name__)

def synthesize_with_improved_thresholds(results: List[Dict[str, Any]], query: str, max_tokens: int = 1800) -> Dict[str, Any]:
    """
    Synthesize answer with special handling for resume content
    """
    try:
        logger.info(f"=== SYNTHESIS DEBUG === Query: '{query}', Input results: {len(results)}")
        
        if not results:
            return {"answer": "I couldn't find relevant information in the uploaded documents.", "sources": []}
        
        # Log what content we're working with
        for i, result in enumerate(results[:3]):
            similarity = result.get('meta', {}).get('similarity', 0)
            source = result.get('meta', {}).get('source', 'unknown')
            content_preview = result.get('content', '')[:100].replace('\n', ' ') + '...'
            logger.info(f"Synthesis input {i+1}: similarity={similarity:.4f}, source='{source}', content='{content_preview}'")
        
        # Detect resume queries
        query_lower = query.lower()
        resume_keywords = ['skill', 'experience', 'project', 'education', 'resume', 'cv', 'qualification']
        is_resume_query = any(keyword in query_lower for keyword in resume_keywords)
        
        # Set very low threshold for resume queries
        if is_resume_query:
            similarity_threshold = 0.01  # Almost no filtering for resume content
            logger.info(f"Resume query detected - using threshold: {similarity_threshold}")
        else:
            similarity_threshold = 0.12
            logger.info(f"Regular query - using threshold: {similarity_threshold}")
        
        # Filter results
        filtered_results = [
            r for r in results 
            if r.get('meta', {}).get('similarity', 0) >= similarity_threshold
        ]
        
        logger.info(f"After threshold filtering: {len(filtered_results)} results")
        
        # If no results pass threshold, use all available results for resume queries
        if not filtered_results and is_resume_query:
            filtered_results = results
            logger.info("No results passed threshold - using all results for resume query")
        elif not filtered_results:
            filtered_results = sorted(results, key=lambda x: x.get('meta', {}).get('similarity', 0), reverse=True)[:2]
            logger.info("No results passed threshold - using top 2 results")
        
        # Prepare context
        context_parts = []
        used_sources = []
        
        for i, chunk in enumerate(filtered_results):
            content = chunk.get("content", "").strip()
            if content:
                context_parts.append(content)
                source_info = {
                    "source": chunk.get("meta", {}).get("source", "unknown"),
                    "similarity": chunk.get("meta", {}).get("similarity", 0)
                }
                used_sources.append(source_info)
        
        if not context_parts:
            return {"answer": "I couldn't find relevant information in the uploaded documents.", "sources": []}
        
        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Context length: {len(context)} characters")
        
        # Create system prompt optimized for resume content
        if is_resume_query:
            system_prompt = (
                "You are analyzing resume/CV content. Follow these rules:\n"
                "1. ONLY extract information that is explicitly mentioned in the provided text\n"
                "2. For skills questions, list the exact technical skills, programming languages, tools, or technologies mentioned\n"
                "3. Do not add any general knowledge or assumptions\n"
                "4. If the information is not in the text, say 'Not mentioned in the provided resume'\n"
                "5. Use bullet points for lists\n"
                "6. Be precise and factual"
            )
        else:
            system_prompt = (
                "You are a helpful assistant. Answer based only on the provided context.\n"
                "If the information is not in the context, say 'I don't have enough information'.\n"
                "Use bullet points for lists and be precise."
            )
        
        # Import the client from the original synthesis module
        from .synthesis import client
        
        # Get completion from GROQ
        try:
            logger.info("Calling GROQ API...")
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Updated model (70b-versatile is decommissioned)
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
                ],
                temperature=0.1,  # Very low temperature for factual extraction
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            answer = completion.choices[0].message.content
            if answer is None:
                return {"answer": "Error: No response generated", "sources": used_sources}
            
            answer = answer.strip()
            logger.info(f"Generated answer length: {len(answer)} characters")
            
            return {"answer": answer, "sources": used_sources}
            
        except Exception as gpt_error:
            logger.error(f"GROQ API error: {str(gpt_error)}")
            # Fallback: return the context directly with some formatting
            fallback_answer = f"Based on the uploaded documents:\n\n{context}"
            return {"answer": fallback_answer, "sources": used_sources}
        
    except Exception as e:
        logger.error(f"Error in synthesis: {str(e)}")
        return {"answer": "Error processing your query.", "sources": []}
