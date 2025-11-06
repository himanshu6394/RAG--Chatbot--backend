"""
Text chunking service for RAG pipeline
"""
import re
import logging
from typing import List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

def chunk_text(text: str, max_chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
    """
    Split text into overlapping chunks of approximately equal size.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Use settings defaults if not provided
    if max_chunk_size is None:
        max_chunk_size = getattr(settings, 'MAX_CHUNK_SIZE', 800)
    if chunk_overlap is None:
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 250)
    
    # Ensure they are positive integers and help type checker
    assert max_chunk_size is not None
    assert chunk_overlap is not None
    max_chunk_size = max(100, max_chunk_size)
    chunk_overlap = max(0, chunk_overlap)
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    start = 0
    text_length = len(text)
    
    logger.info(f"Chunking text of length {text_length} with chunk_size={max_chunk_size}, overlap={chunk_overlap}")
    
    while start < text_length:
        # Calculate end position
        end = min(start + max_chunk_size, text_length)
        
        # Try to break at sentence boundary if possible
        if end < text_length:
            # Look for sentence endings in the last 200 characters
            search_start = max(end - 200, start)
            sentence_break = -1
            
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in '.!?':
                    sentence_break = i + 1
                    break
            
            if sentence_break > start:
                end = sentence_break
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 10:  # Only add chunks with meaningful content
            chunks.append(chunk)
            logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")
        
        # Move start position with overlap
        start = max(end - chunk_overlap, start + 1)
        
        # Prevent infinite loop
        if start >= text_length:
            break
    
    logger.info(f"Created {len(chunks)} chunks from input text")
    return chunks

def smart_chunk_text(text: str, max_chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
    """
    Advanced chunking that preserves semantic boundaries, especially for resumes
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Use settings defaults if not provided
    if max_chunk_size is None:
        max_chunk_size = getattr(settings, 'MAX_CHUNK_SIZE', 800)
    if chunk_overlap is None:
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 250)
    
    # Ensure they are positive integers and help type checker
    assert max_chunk_size is not None
    assert chunk_overlap is not None
    max_chunk_size = max(100, max_chunk_size)
    chunk_overlap = max(0, chunk_overlap)
    
    # Special handling for resume content
    if any(term in text.lower() for term in ['technical skills', 'programming languages', 'experience', 'education', 'projects']):
        logger.info("Detected resume content - using resume-specific chunking")
        return chunk_resume_content(text, max_chunk_size, chunk_overlap)
    
    # Regular chunking for other content
    return chunk_regular_content(text, max_chunk_size, chunk_overlap)

def chunk_resume_content(text: str, max_chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Special chunking for resume content to keep sections together
    """
    chunks = []
    
    # Common resume section patterns
    section_patterns = [
        r'(technical\s+skills?:?.*?)(?=\n\s*[A-Z][^:]*:|$)',
        r'(programming\s+languages?:?.*?)(?=\n\s*[A-Z][^:]*:|$)', 
        r'(experience:?.*?)(?=\n\s*[A-Z][^:]*:|$)',
        r'(education:?.*?)(?=\n\s*[A-Z][^:]*:|$)',
        r'(projects?:?.*?)(?=\n\s*[A-Z][^:]*:|$)',
        r'(other\s+skills?:?.*?)(?=\n\s*[A-Z][^:]*:|$)'
    ]
    
    used_text = set()
    
    # Extract known sections first
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_text = match.group(1).strip()
            if section_text and len(section_text) > 10:
                # Clean up the section text
                section_text = re.sub(r'\s+', ' ', section_text)
                chunks.append(section_text)
                used_text.add(section_text.lower()[:50])  # Mark as used
    
    # If no sections found, fall back to regular chunking
    if not chunks:
        return chunk_regular_content(text, max_chunk_size, chunk_overlap)
    
    logger.info(f"Resume chunking created {len(chunks)} section-based chunks")
    return chunks

def chunk_regular_content(text: str, max_chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Regular paragraph-based chunking for non-resume content
    """
    # First try to split by paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too long, split it further
            if len(paragraph) > max_chunk_size:
                para_chunks = chunk_text(paragraph, max_chunk_size, chunk_overlap)
                chunks.extend(para_chunks)
                current_chunk = ""
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
