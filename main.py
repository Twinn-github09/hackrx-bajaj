from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import time
import asyncio
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from models.schemas import QueryRequest, QueryResponse, CompetitionResponse, HealthCheck, ErrorResponse
from core.document_loader import DocumentLoaderFactory
from core.document_parser import DocumentParserFactory
from core.embeddings import EmbeddingService
from core.hybrid_processor import HybridQueryProcessor
from utils.logging_config import setup_logging, get_logger
from utils.helpers import timing_decorator, async_retry

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(level=log_level)
logger = get_logger(__name__)

# Global variables for services
embedding_service = None
query_processor = None
security = HTTPBearer()

# Expected bearer token
EXPECTED_TOKEN = os.getenv("BEARER_TOKEN")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting RAG API service...")
    
    global embedding_service, query_processor
    
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
        )
        
        # Initialize query processor with hybrid approach
        query_processor = HybridQueryProcessor(
            embedding_service=embedding_service,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        )
        
        # Initialize all models (this might take some time)
        logger.info("Initializing models...")
        await asyncio.get_event_loop().run_in_executor(None, query_processor.initialize)
        
        logger.info("✅ RAG API service started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API service...")

# Create FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Fast RAG system for processing documents and answering queries",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    global embedding_service, query_processor
    
    models_status = {
        "embedding_service": embedding_service is not None,
        "query_processor": query_processor is not None,
    }
    
    return HealthCheck(
        status="healthy" if all(models_status.values()) else "unhealthy",
        models_loaded=models_status
    )

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_queries_full(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """Full endpoint with detailed response"""
    return await _process_queries_internal(request, include_details=True)

@app.post("/hackrx/run", response_model=CompetitionResponse)
async def process_queries_competition(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> CompetitionResponse:
    """
    Competition endpoint - EXACT format as required
    POST /hackrx/run
    Returns only 'answers' array as specified in competition requirements
    """
    try:
        result = await _process_queries_internal(request, include_details=False)
        # Ensure we return exactly the format required: {"answers": [...]}
        return CompetitionResponse(answers=result.answers)
    except Exception as e:
        logger.error(f"Competition endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed"
        )
@timing_decorator
@async_retry(max_retries=2, delay=1.0)
async def _process_queries_internal(
    request: QueryRequest,
    include_details: bool = True
) -> QueryResponse:
    """
    Main endpoint for processing document queries
    
    This endpoint:
    1. Downloads/loads the document
    2. Parses and chunks the content
    3. Creates embeddings and stores in vector database
    4. Processes each question using RAG pipeline
    5. Returns structured answers with reasoning
    """
    start_time = time.time()
    
    try:
        global embedding_service, query_processor
        
        if not embedding_service or not query_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Services not initialized. Please check system health."
            )
        
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # 1. Load document
        document_loader = DocumentLoaderFactory.create_loader(request.documents)
        document_path = await document_loader.load(request.documents)
        
        # 2. Parse document
        parser = DocumentParserFactory.create_parser(document_path)
        parsed_doc = await parser.parse(document_path)
        
        logger.info(f"Parsed document with {len(parsed_doc.get('content', []))} sections")
        
        # 3. Process document for embeddings with improved parameters
        chunks_count = await asyncio.get_event_loop().run_in_executor(
            None,
            embedding_service.process_document,
            parsed_doc['total_text'],
            parsed_doc['metadata']
        )
        
        logger.info(f"Created {chunks_count} chunks and embeddings")
        
        # 4. Process each question
        detailed_results = []
        answers = []
        
        for question in request.questions:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Process query using enhanced hybrid RAG pipeline
            result = await query_processor.process_query(
                question,
                int(os.getenv("TOP_K_RETRIEVAL", 20)),
                int(os.getenv("RERANK_TOP_K", 8)),
                float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
            )
            
            detailed_results.append(result)
            answers.append(result.answer)
            
            logger.info(f"Answered with confidence: {result.confidence:.3f}")
        
        total_time = time.time() - start_time
        
        logger.info(f"✅ Completed processing {len(request.questions)} questions in {total_time:.2f}s")
        
        return QueryResponse(
            answers=answers,
            detailed_results=detailed_results,
            total_processing_time=total_time,
            document_metadata=parsed_doc['metadata']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/query", response_model=QueryResponse)
async def single_query(
    question: str,
    document_url: str,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """
    Simplified endpoint for single question processing
    """
    request = QueryRequest(
        documents=document_url,
        questions=[question]
    )
    
    return await _process_queries_internal(request)

@app.get("/api/v1/models", response_model=dict)
async def get_model_info(token: str = Depends(verify_token)):
    """Get information about loaded models"""
    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "llm_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "llm_provider": "Groq",
        "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "vector_store": "FAISS",
        "status": "loaded" if query_processor else "not_loaded",
        "api_based": True
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return ErrorResponse(
        error=f"HTTP {exc.status_code}",
        detail=exc.detail
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return ErrorResponse(
        error="Internal Server Error",
        detail="An unexpected error occurred"
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 1))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level=log_level.lower()
    )
