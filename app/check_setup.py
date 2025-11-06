import os
import sys
import platform
from pathlib import Path
import logging
from app.services.ocr_service import find_tesseract, find_poppler
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables and paths are set up correctly."""
    try:
        settings = get_settings()

        # Check GROQ API key
        if not os.getenv('GROQ_API_KEY'):
            logger.error("GROQ_API_KEY environment variable is not set")
            return False
            
        # Check directories
        dirs_to_check = [
            settings.UPLOAD_DIR,
            settings.CHROMA_DB_PATH,
        ]
        
        for dir_path in dirs_to_check:
            path = Path(dir_path)
            if not path.exists():
                logger.info(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
        
        # Check OCR dependencies
        tesseract_path = find_tesseract()
        if not tesseract_path:
            logger.error("Tesseract not found. Please install Tesseract OCR")
            return False
            
        poppler_path = find_poppler()
        if not poppler_path and platform.system() == "Windows":
            logger.error("Poppler not found on Windows. Please install Poppler")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error during environment check: {str(e)}")
        return False

if __name__ == "__main__":
    if check_environment():
        logger.info("All environment checks passed successfully!")
        sys.exit(0)
    else:
        logger.error("Environment check failed. Please fix the issues above.")
        sys.exit(1)
