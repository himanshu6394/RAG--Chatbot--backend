import os
import sys
import platform
import pytesseract
from pdf2image.pdf2image import convert_from_bytes, convert_from_path
from PyPDF2 import PdfReader
from typing import List, Union
import logging
from pathlib import Path
import re

# Configure logging with proper format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure logging
logger = logging.getLogger(__name__)

# Configure Tesseract path based on OS
def find_tesseract():
    # First check environment variable
    tesseract_path = os.getenv('TESSERACT_PATH')
    if tesseract_path and os.path.exists(tesseract_path):
        return tesseract_path

    # Common Windows install paths
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    
    if platform.system() == "Windows":
        # Try common Windows paths
        for path in windows_paths:
            if os.path.exists(path):
                return path
                
        # Check if it's in PATH
        from shutil import which
        system_tesseract = which('tesseract')
        if system_tesseract:
            return system_tesseract
            
        logger.warning("Tesseract not found in common locations. Please install Tesseract or set TESSERACT_PATH environment variable.")
        return None

# Set Tesseract path
tesseract_executable = find_tesseract()
if tesseract_executable:
    pytesseract.pytesseract.tesseract_cmd = tesseract_executable
    logger.info(f"Using Tesseract from: {tesseract_executable}")
else:
    logger.warning("Tesseract path not configured. OCR functionality will not work.")

# Configure poppler path for Windows (needed by pdf2image)
def find_poppler():
    # First check environment variable
    poppler_path = os.getenv('POPPLER_PATH')
    if poppler_path and os.path.exists(poppler_path):
        return poppler_path

    # Common Windows install paths
    windows_paths = [
        r"C:\Program Files\Release-25.07.0-0\poppler-25.07.0\Library\bin",
        r"C:\Program Files\poppler-23.11.0\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler\bin",
        r"C:\Program Files (x86)\poppler\bin"
    ]
    
    if platform.system() == "Windows":
        # Try common Windows paths
        for path in windows_paths:
            if os.path.exists(path):
                return path
                
        logger.warning("Poppler not found in common locations. Please install Poppler or set POPPLER_PATH environment variable.")
        return None

POPPLER_PATH = find_poppler()
if POPPLER_PATH:
    logger.info(f"Using Poppler from: {POPPLER_PATH}")
else:
    logger.warning("Poppler path not configured. PDF to image conversion will not work.")

def extract_text_from_file(file_path: str) -> Union[str, List[dict]]:
    """
    Extracts text from a file on disk (PDF, image, or text).

    Supports the following:
    - PDFs with selectable text
    - PDFs requiring OCR (if no text found)
    - PNG, JPG, JPEG images using Tesseract OCR
    - Plain text files (UTF-8 encoded)

    Args:
        file_path (str): Path to the file on disk

    Returns:
        Union[str, List[dict]]: Either the extracted text as a string or a list of chunks with metadata
    """
    try:
        filename = os.path.basename(file_path).lower()

        if filename.endswith(".pdf"):
            try:
                # First try to extract text directly from PDF
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    text = ""
                    total_pages = len(reader.pages)
                    
                    logger.info(f"Processing PDF: {filename} with {total_pages} pages")
                    
                    for page_num, page in enumerate(reader.pages, 1):
                        logger.info(f"Processing page {page_num}/{total_pages}")
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"

                    if not text.strip():  # fallback to OCR if no text found
                        logger.info(f"No text found in PDF {filename}, falling back to OCR")
                        try:
                            # Use poppler path on Windows
                            if platform.system() == "Windows" and POPPLER_PATH and os.path.exists(POPPLER_PATH):
                                logger.info(f"Converting PDF to images using Poppler at {POPPLER_PATH}")
                                images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
                                # Process each page
                                for i, image in enumerate(images, start=1):
                                    page_text = pytesseract.image_to_string(image)
                                    text += f"\n--- Page {i} ---\n{page_text}"
                            else:
                                logger.warning("Skipping OCR for PDF - Poppler not configured correctly")
                        except Exception as ocr_error:
                            logger.error(f"OCR failed for PDF {filename}: {str(ocr_error)}")
                            return f"PDF OCR failed: {str(ocr_error)}"
                    return text.strip()

            except Exception as pdf_error:
                logger.error(f"PDF processing failed for {filename}: {str(pdf_error)}")
                return f"PDF processing failed: {str(pdf_error)}"

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                logger.info(f"Processing image file: {filename}")
                # Check if tesseract is properly configured
                if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                    error_msg = "Tesseract not found. Please install Tesseract OCR and set the correct path."
                    logger.error(error_msg)
                    return error_msg

                # Try OCR with error handling
                try:
                    text = pytesseract.image_to_string(file_path)
                    if not text.strip():
                        return "No text could be extracted from the image"
                    return text.strip()
                except pytesseract.TesseractError as te:
                    logger.error(f"Tesseract error for {filename}: {str(te)}")
                    return f"OCR processing error: {str(te)}"
                    
            except Exception as img_error:
                logger.error(f"Image OCR failed for {filename}: {str(img_error)}")
                return f"Image OCR failed: {str(img_error)}"

        elif filename.endswith((".txt", ".md", ".rst", ".doc", ".docx")):
            try:
                logger.info(f"Processing text file: {filename}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                return text.strip()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {filename}, trying with different encodings")
                encodings = ['latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            text = file.read()
                        return text.strip()
                    except UnicodeDecodeError:
                        continue
                logger.error(f"All encoding attempts failed for {filename}")
                return "Failed to decode text file with multiple encodings"
            except Exception as text_error:
                logger.error(f"Text file processing failed for {filename}: {str(text_error)}")
                return f"Text processing failed: {str(text_error)}"
        else:
            return f"Unsupported file type: {filename}"

    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {str(e)}")
        return f"Text extraction failed: {str(e)}"

def chunk_text(text: str, max_tokens: int = 800) -> List[dict]:
    """
    Splits a large block of text into smaller paragraph-based chunks.
    Converts every new line in a paragraph to a bullet point for cleaner downstream responses.
    """
    paragraphs = text.split("\n\n")
    cleaned_chunks = []
    for p in paragraphs:
        # Replace all newlines with bullet points
        bulletified = '\n'.join([f'- {line.strip()}' for line in p.strip().split('\n') if line.strip()])
        if bulletified:
            cleaned_chunks.append({"content": bulletified, "meta": {"source": "upload"}})
    return cleaned_chunks
