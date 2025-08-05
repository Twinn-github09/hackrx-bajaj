from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class QueryRequest(BaseModel):
    """Request model for query processing"""
    documents: str = Field(..., description="Document URL or path")
    questions: List[str] = Field(..., description="List of questions to answer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "Does this policy cover maternity expenses?"
                ]
            }
        }

class DocumentChunk(BaseModel):
    """Model for document chunks with enhanced scoring"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = {}
    page_number: Optional[int] = None
    chunk_index: int
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None  # Added for reranking
    
    def copy(self):
        """Create a copy of the chunk"""
        return DocumentChunk(
            id=self.id,
            content=self.content,
            metadata=self.metadata.copy(),
            page_number=self.page_number,
            chunk_index=self.chunk_index,
            similarity_score=self.similarity_score,
            rerank_score=self.rerank_score
        )

class RetrievalResult(BaseModel):
    """Model for retrieval results"""
    chunks: List[DocumentChunk]
    query: str
    total_chunks: int
    retrieval_time: float

class AnswerResult(BaseModel):
    """Model for individual answer result"""
    question: str
    answer: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str
    source_chunks: List[DocumentChunk] = []
    processing_time: float

class QueryResponse(BaseModel):
    """Response model for query processing"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")
    detailed_results: Optional[List[AnswerResult]] = None
    total_processing_time: float
    document_metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment.",
                    "Yes, the policy covers maternity expenses with conditions."
                ],
                "total_processing_time": 2.5
            }
        }

class CompetitionResponse(BaseModel):
    """Competition-specific response model - ONLY answers field as required"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment.",
                    "Yes, the policy covers maternity expenses with conditions."
                ]
            }
        }

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    models_loaded: Dict[str, bool] = {}

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
