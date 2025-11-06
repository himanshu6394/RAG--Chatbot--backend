#!/usr/bin/env python3
"""
Script to clear the vector database and re-upload all documents with improved chunking
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.services.vector_store import clear_vector_store, store_text_chunks
from app.services.ocr_service import extract_text_from_file
from app.services.chunking import smart_chunk_text
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reprocess_documents():
    """Clear database and re-upload all documents with improved chunking"""
    try:
        # Clear the vector store
        logger.info("Clearing vector store...")
        result = clear_vector_store()
        if result.get("status") != "success":
            logger.error(f"Failed to clear vector store: {result}")
            return False
        
        # Get uploads directory
        uploads_dir = Path(backend_dir) / "uploads"
        if not uploads_dir.exists():
            logger.error("Uploads directory not found!")
            return False
        
        # Process all files in uploads directory
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error("No PDF files found in uploads directory!")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            
            try:
                # Extract text
                extracted_text = extract_text_from_file(str(pdf_file))
                if not extracted_text:
                    logger.error(f"Failed to extract text from {pdf_file.name}")
                    continue
                
                # Convert to string if needed
                if isinstance(extracted_text, list):
                    text_content = " ".join(str(item) for item in extracted_text)
                else:
                    text_content = str(extracted_text)
                
                logger.info(f"Extracted {len(text_content)} characters from {pdf_file.name}")
                
                # Use smart chunking
                chunk_texts = smart_chunk_text(
                    text_content,
                    max_chunk_size=getattr(settings, 'MAX_CHUNK_SIZE', 800),
                    chunk_overlap=getattr(settings, 'CHUNK_OVERLAP', 250)
                )
                
                if not chunk_texts:
                    logger.error(f"No chunks created for {pdf_file.name}")
                    continue
                
                logger.info(f"Created {len(chunk_texts)} chunks from {pdf_file.name}")
                
                # Format chunks for storage
                formatted_chunks = []
                for i, chunk_text in enumerate(chunk_texts):
                    formatted_chunks.append({
                        'content': chunk_text,
                        'meta': {
                            'source': pdf_file.name,
                            'chunk_index': i,
                            'total_chunks': len(chunk_texts),
                            'reprocessed': True
                        }
                    })
                
                # Store in vector database
                storage_result = store_text_chunks(formatted_chunks)
                
                if storage_result.get("status") == "success":
                    logger.info(f"Successfully stored {len(formatted_chunks)} chunks from {pdf_file.name}")
                else:
                    logger.error(f"Failed to store chunks from {pdf_file.name}: {storage_result}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                continue
        
        logger.info("Document reprocessing completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in reprocess_documents: {str(e)}")
        return False

if __name__ == "__main__":
    success = reprocess_documents()
    if success:
        print("✅ Successfully reprocessed all documents with improved chunking!")
    else:
        print("❌ Failed to reprocess documents. Check logs for details.")
