import uvicorn
import sys
from pathlib import Path
import logging
from app.check_setup import check_environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    
    # Check environment setup
    logger.info("Checking environment setup...")
    if not check_environment():
        logger.error("Environment setup check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Start the FastAPI server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
