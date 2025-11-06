import os
import logging
import re
from typing import List, Dict, Union, Any, Optional
from dotenv import load_dotenv
from groq import Groq

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize GROQ client
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    client = Groq(api_key=api_key)
except Exception as e:
    logger.error(f"Error initializing GROQ client: {str(e)}")
    raise

# Type alias for chunk structure
ChunkType = Dict[str, Any]

def format_response(text: str) -> str:
    """
    Format the response text to ensure proper line breaks and readability.
    Also removes any <think>...</think> blocks from LLM output.
    Ensures every new point starts with a bullet point.
    
    Args:
        text: The text to format
        
    Returns:
        str: Formatted text with proper line breaks
    """
    if not text:
        return text
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Split into sentences and clean up
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    # Add line breaks between sentences
    formatted = "\n".join(sentences)
    # Ensure bullet points are on new lines
    formatted = re.sub(r'(?<!\n)•', '\n•', formatted)
    # Add bullet point to each new line if not already present
    formatted = '\n'.join([f'- {line.lstrip("- ")}' if not line.startswith('-') and line else line for line in formatted.split('\n')])
    # Clean up multiple newlines
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    return formatted.strip()

def summarize_themes(results: List[ChunkType], query: str, max_tokens: int = 800) -> Dict[str, Any]:
    """
    Synthesize search results into a coherent answer using GPT.
    Only send a safe number of chunks/tokens to the LLM to avoid API errors.
    Implements adaptive similarity threshold and improved context handling.
    
    Args:
        results: List of search results with content and metadata
        query: The original user query
        max_tokens: Maximum tokens for LLM response (default 300)
        
    Returns:
        dict: {"answer": str, "sources": List[Dict[str, Any]]}
    """
    try:
        if not results:
            return {"answer": "No relevant information found.", "sources": []}
        # Improved adaptive similarity threshold
        if len(results) > 10:
            similarity_threshold = 0.35
        else:
            similarity_threshold = 0.2
        top_n = 10
        filtered_results = [r for r in results if r.get('meta', {}).get('similarity', 0) >= similarity_threshold]
        if not filtered_results:
            return {"answer": "No relevant information found above similarity threshold.", "sources": []}
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.get('meta', {}).get('similarity', 0),
            reverse=True
        )
        # Truncate context to fit within max tokens
        context_parts = []
        total_tokens = 0
        for chunk in sorted_results[:top_n]:
            content = chunk.get('content', '')
            # Estimate tokens (roughly 1 token per 4 chars)
            chunk_tokens = max(len(content) // 4, 1)
            if total_tokens + chunk_tokens > max_tokens:
                # Truncate content to fit remaining tokens
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 0:
                    max_chars = remaining_tokens * 4
                    truncated_content = content[:max_chars]
                    context_parts.append(truncated_content)
                    total_tokens += remaining_tokens
                break
            else:
                context_parts.append(content)
                total_tokens += chunk_tokens
        if not context_parts:
            return {"answer": "Could not process any of the search results.", "sources": []}
        context = "\n\n".join(context_parts)
        # Only include sources for used chunks
        used_sources = [
            {"source": chunk.get("meta", {}).get("source", "unknown"), "similarity": chunk.get("meta", {}).get("similarity", 0)}
            for chunk in sorted_results[:top_n]
        ]
        # Create system prompt
        system_prompt = (
            "You are a document analysis assistant. STRICT REQUIREMENTS:\n"
            "1. ONLY answer using the exact information from the provided context.\n"
            "2. If the specific information requested is NOT found in the context, respond: 'This information is not available in the provided documents.'\n"
            "3. DO NOT make assumptions or provide general knowledge.\n"
            "4. DO NOT mix information from different documents unless specifically asked.\n"
            "5. For skills/resume queries: Only mention skills explicitly listed in resume documents.\n"
            "6. For academic/literature queries: Only use information from academic documents.\n"
            "7. If context contains unrelated information, ignore it and state information is not available."
        )
        # Get completion from GROQ
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Using more conservative model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based ONLY on the context provided below, answer this question: {query}\n\nCONTEXT (use only this information):\n{context}\n\nRemember: If the answer is not in the context above, say 'This information is not available in the provided documents.'"}
                ],
                temperature=0.1,  # Very low temperature for more deterministic responses
                max_tokens=max_tokens,
                presence_penalty=0.5,  # Higher penalty to avoid generic responses
                frequency_penalty=0.5,  # Higher penalty to avoid repetitive responses
                top_p=0.8,           # Lower top_p for more focused responses
                n=1,
                stop=None
            )
            
            answer = completion.choices[0].message.content
            if answer is None:
                return {"answer": "Error: No response generated", "sources": used_sources}
                
            # Format the response with proper line breaks and clean <think> blocks
            return {"answer": format_response(answer), "sources": None}
            
        except Exception as gpt_error:
            logger.error(f"Error calling OpenAI API: {str(gpt_error)}")
            # Fallback to a simpler response using the most relevant chunk
            if context_parts:
                return {"answer": format_response("API Error. Here's the most relevant excerpt."), "sources": used_sources}
            return {"answer": "Error generating response and no fallback content available.", "sources": []}
        
    except Exception as e:
        logger.error(f"Error in summarize_themes: {str(e)}")
        return {"answer": f"Error generating summary: {str(e)}", "sources": []}



