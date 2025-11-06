import os
import hashlib
import logging
import asyncio
import re
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ...services.ocr_service import extract_text_from_file
from ...services.vector_store import store_text_chunks, get_stored_documents
from ...services.chunking import smart_chunk_text
from app.core.config import settings  # Correct import for settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

upload_router = APIRouter()

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx', '.jpg', '.png'}
CHUNK_SIZE = 800  # Use fixed chunk size for all files

def is_valid_file(file: UploadFile) -> bool:
    """
    Validate file extension and size.
    """
    try:
        if not file.filename:
            return False
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating file {file.filename}: {str(e)}")
        return False

def get_file_hash(content: bytes) -> str:
    """
    Generate SHA-256 hash of file content to prevent duplicates.
    """
    return hashlib.sha256(content).hexdigest()

def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = 20) -> List[Dict[str, Any]]:
    """
    Splits extracted text into sentence-based chunks with overlap for better semantic retrieval.
    Each chunk is ~chunk_size words, but never splits a sentence. Overlap is in sentences.
    """
    try:
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        chunk_index = 0
        for para in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?]) +', para)
            current_chunk = []
            current_len = 0
            i = 0
            while i < len(sentences):
                sentence = sentences[i]
                words = sentence.split()
                if current_len + len(words) <= chunk_size or not current_chunk:
                    current_chunk.append(sentence)
                    current_len += len(words)
                    i += 1
                else:
                    chunk = " ".join(current_chunk)
                    chunk_dict = {
                        "content": chunk,
                        "meta": {
                            "source": source,
                            "chunk_index": chunk_index,
                            "total_chunks": None  # Set later
                        }
                    }
                    chunks.append(chunk_dict)
                    chunk_index += 1
                    # Overlap: start next chunk with last N sentences
                    current_chunk = current_chunk[-2:] if 2 > 0 else []
                    current_len = sum(len(s.split()) for s in current_chunk)
            # Add last chunk in paragraph
            if current_chunk:
                chunk = " ".join(current_chunk)
                chunk_dict = {
                    "content": chunk,
                    "meta": {
                        "source": source,
                        "chunk_index": chunk_index,
                        "total_chunks": None
                    }
                }
                chunks.append(chunk_dict)
                chunk_index += 1
        # Set total_chunks meta
        for c in chunks:
            c["meta"]["total_chunks"] = len(chunks)
        # Log only summary info, not content
        if chunks:
            logger.info(f"Chunked {source}: {len(chunks)} chunks. First chunk preview: '{chunks[0]['content'][:20]}...'")
        else:
            logger.info(f"Chunked {source}: No chunks created.")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text from {source}: {str(e)}")
        raise ValueError(f"Failed to chunk text: {str(e)}")

async def process_single_file(file: UploadFile) -> Dict[str, Any]:
    """
    Process a single file and return its results.
    """
    try:
        if not file.filename:
            return {
                "filename": "unknown",
                "status": "failed",
                "error": "No filename provided"
            }

        if not is_valid_file(file):
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            }

        # Generate unique filename using hash
        file_hash = get_file_hash(content)
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{file_hash}{ext}"
        file_path = os.path.join("uploads", unique_filename)

        # Save file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(content)

        # Extract text
        logger.info(f"Extracting text from {file.filename}")
        extracted = extract_text_from_file(file_path)

        if not extracted or len(extracted) == 0 or "failed" in str(extracted).lower():
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"Text extraction failed: {extracted}"
            }

        # Use the new smart chunking service
        if isinstance(extracted, str):
            text_content = extracted
        elif isinstance(extracted, list):
            text_content = " ".join(str(item) for item in extracted)
        else:
            return {
                "filename": file.filename,
                "status": "failed",
                "error": "Unsupported return type from text extraction"
            }

        # Create chunks using the improved chunking service
        chunk_texts = smart_chunk_text(
            text_content,
            max_chunk_size=getattr(settings, 'MAX_CHUNK_SIZE', 800),
            chunk_overlap=getattr(settings, 'CHUNK_OVERLAP', 250)
        )

        if not chunk_texts:
            return {
                "filename": file.filename,
                "status": "failed",
                "error": "No text could be extracted or chunked from the document"
            }

        # Format chunks for storage
        formatted_chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            formatted_chunks.append({
                'content': chunk_text,
                'meta': {
                    'source': file.filename,
                    'chunk_index': i,
                    'total_chunks': len(chunk_texts),
                    'file_hash': file_hash
                }
            })

        # Store chunks in vector database
        storage_result = store_text_chunks(formatted_chunks)
        
        if storage_result.get("status") == "success":
            logger.info(f"Successfully processed {file.filename}: {len(formatted_chunks)} chunks stored in vector database")
            return {
                "filename": file.filename,
                "status": "success",
                "chunks": len(formatted_chunks),
                "message": f"Successfully stored {len(formatted_chunks)} chunks",
                "file_hash": file_hash,
                "storage_result": storage_result
            }
        else:
            logger.error(f"Failed to store chunks for {file.filename}: {storage_result}")
            return {
                "filename": file.filename,
                "status": "failed",
                "error": f"Failed to store document chunks: {storage_result.get('message', 'Unknown error')}",
                "chunks": len(formatted_chunks)
            }

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return {
            "filename": file.filename,
            "status": "failed",
            "error": str(e)
        }

@upload_router.post("/")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple files simultaneously.
    Returns a summary of the processing results for each file.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Process files concurrently
        tasks = [process_single_file(file) for file in files]
        results = await asyncio.gather(*tasks)

        # Calculate totals
        total_chunks = sum(
            result["chunks"] 
            for result in results 
            if result["status"] == "success"
        )

        successful_files = sum(1 for result in results if result["status"] == "success")
        failed_files = sum(1 for result in results if result["status"] == "failed")

        return {
            "status": "completed",
            "total_files": len(files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "results": results
        }

    except Exception as ex:
        logger.error(f"Error in upload endpoint: {str(ex)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"An error occurred while processing the uploaded documents: {str(ex)}",
                "results": []
            }
        )

@upload_router.get("/status")
async def get_upload_status():
    """Get information about uploaded documents in the vector database"""
    try:
        return get_stored_documents()
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        return {"error": str(e)}
