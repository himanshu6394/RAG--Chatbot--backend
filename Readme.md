# Document Chatbot

A sophisticated document-based chatbot system that processes PDF documents and images, extracts text, and provides intelligent responses to user queries based on the document content.

## 🌟 Features

- **Document Processing**:

  - PDF text extraction (both searchable and scanned PDFs)
  - Image OCR support (PNG, JPG, JPEG)
  - Plain text file support
  - Automatic text chunking for efficient processing

- **Intelligent Search**:

  - Vector-based semantic search using SentenceTransformers
  - Contextual response generation
  - Source-specific querying capabilities

- **Advanced Text Processing**:
  - Text summarization using T5 transformer model
  - Theme extraction and synthesis
  - Metadata preservation

## 🏗️ System Architecture

### Components

1. **FastAPI Backend**:

   - RESTful API endpoints for document upload and querying
   - CORS middleware for cross-origin requests
   - Modular router structure

2. **Document Processing Pipeline**:

   - OCR Service (`ocr_service.py`)
   - Vector Store Service (`vector_store.py`)
   - Text Synthesis Service (`synthesis.py`)

3. **Database**:
   - ChromaDB for vector storage
   - Persistent storage for embeddings

### Data Flow

1. **Document Upload Flow**:

   ```
   Client -> Upload API -> OCR Service -> Text Chunking -> Vector Store
   ```

2. **Query Flow**:
   ```
   Client -> Query API -> Vector Search -> Text Synthesis -> Response
   ```

## 🛠️ Technical Stack

- **Backend Framework**: FastAPI
- **OCR Engine**: Tesseract (via pytesseract)
- **PDF Processing**: PyPDF2, pdf2image
- **Vector Store**: ChromaDB
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Text Summarization**: Hugging Face Transformers (t5-base)

## 📋 Requirements

- Python 3.x
- Tesseract OCR engine
- Required Python packages (see requirements.txt)
- OpenAI API key (for enhanced query processing)

## 🚀 Setup & Installation

1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # For Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Tesseract OCR engine
5. Set up environment variables:

   - Create a .env file
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

6. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## 🔌 API Endpoints

1. **Upload Document**

   - Endpoint: `/upload`
   - Method: POST
   - Accepts: PDF, PNG, JPG, JPEG, TXT files

2. **Query Documents**

   - Endpoint: `/query`
   - Method: GET
   - Parameters:
     - q: Query string
     - source: (Optional) Specific document source

3. **Theme Analysis**
   - Endpoint: `/theme`
   - Method: GET

## 💡 Possible Enhancements

1. **Document Processing**:

   - Support for more document formats (DOCX, RTF, etc.)
   - Advanced OCR preprocessing for better accuracy
   - Parallel processing for large documents
   - Document version control

2. **Search & Retrieval**:

   - Implement semantic caching
   - Add fuzzy search capabilities
   - Support for multiple languages
   - Question-answering chain optimization

3. **User Experience**:

   - Real-time processing status updates
   - Document preview functionality
   - Chat history persistence
   - User authentication and document access control

4. **Architecture**:

   - Implement queue system for large document processing
   - Add Redis cache for frequent queries
   - Document chunking optimization
   - Automated testing suite

5. **Analytics**:

   - Query tracking and analysis
   - Document usage statistics
   - Performance metrics dashboard
   - Error tracking and reporting

6. **Security**:
   - Document encryption at rest
   - Rate limiting
   - API key authentication
   - Role-based access control

## 📝 License

MIT License

## 👥 Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request
