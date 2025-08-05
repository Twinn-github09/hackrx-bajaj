"""
Hybrid Query Processor - Combines local models with Groq API for optimal accuracy and reliability
"""
from typing import List, Dict, Any, Optional
import os
import asyncio
import time
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import CrossEncoder
import torch

from models.schemas import DocumentChunk, AnswerResult
from core.groq_processor import GroqLLM

logger = logging.getLogger(__name__)

class LocalQAModel:
    """Local question-answering model using HuggingFace transformers"""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.max_length = 512
        
    def initialize(self):
        """Initialize the local QA model"""
        if self.pipeline is None:
            logger.info(f"Loading local QA model: {self.model_name}")
            self.pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Local QA model loaded successfully")
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer question using local model"""
        if self.pipeline is None:
            self.initialize()
        
        # Truncate context if too long
        if len(context) > 4000:  # Conservative limit
            context = context[:4000]
        
        try:
            result = self.pipeline(question=question, context=context)
            return {
                "answer": result["answer"],
                "confidence": result["score"],
                "start": result.get("start", 0),
                "end": result.get("end", len(result["answer"]))
            }
        except Exception as e:
            logger.error(f"Local QA model error: {str(e)}")
            return {
                "answer": "Unable to extract answer from the provided context.",
                "confidence": 0.0,
                "start": 0,
                "end": 0
            }

class AdvancedReranker:
    """Advanced reranking using cross-encoder model"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        
    def initialize(self):
        """Initialize the reranker model"""
        if self.model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded successfully")
    
    def rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Rerank chunks using cross-encoder"""
        if self.model is None:
            self.initialize()
        
        if not chunks:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, chunk.content) for chunk in chunks]
        
        try:
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Update chunks with reranking scores
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
            
            # Sort by reranking score
            reranked = sorted(chunks, key=lambda x: x.rerank_score or 0, reverse=True)
            
            logger.info(f"Reranked {len(chunks)} chunks, top score: {reranked[0].rerank_score:.3f}")
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            return chunks

class HybridQueryProcessor:
    """Hybrid processor combining Groq API (fast) with local models (reliable)"""
    
    def __init__(self, embedding_service, groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant"):
        self.embedding_service = embedding_service
        self.groq_llm = GroqLLM(groq_api_key, groq_model) if groq_api_key else None
        self.local_qa = LocalQAModel()
        self.reranker = AdvancedReranker()
        self.fallback_mode = False
        
    def initialize(self):
        """Initialize all models"""
        logger.info("Initializing hybrid query processor...")
        
        # Always initialize local models
        self.local_qa.initialize()
        self.reranker.initialize()
        
        # Try to initialize Groq
        if self.groq_llm:
            try:
                self.groq_llm.initialize()
                logger.info("Groq API initialized successfully")
            except Exception as e:
                logger.warning(f"Groq API initialization failed: {str(e)}, using local-only mode")
                self.groq_llm = None
        
        logger.info("Hybrid processor initialized")
    
    async def process_query(
        self,
        question: str,
        top_k: int = 20,
        rerank_top_k: int = 8,
        similarity_threshold: float = 0.5
    ) -> AnswerResult:
        """Process query with hybrid approach"""
        start_time = time.time()
        
        try:
            # 1. Retrieve relevant chunks
            chunks = await self._retrieve_chunks(question, top_k, similarity_threshold)
            
            if not chunks:
                return self._no_context_result(question, start_time)
            
            # 2. Advanced reranking
            reranked_chunks = self.reranker.rerank_chunks(question, chunks)
            top_chunks = reranked_chunks[:rerank_top_k]
            
            # 3. Try Groq API first (fast), fallback to local (reliable)
            result = await self._answer_with_fallback(question, top_chunks)
            
            # 4. Enhanced post-processing
            final_result = self._enhance_answer(result, question, top_chunks, start_time)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return AnswerResult(
                question=question,
                answer="I encountered an error while processing this question.",
                confidence=0.0,
                reasoning=f"Processing error: {str(e)}",
                source_chunks=[],
                processing_time=time.time() - start_time
            )
    
    async def _retrieve_chunks(self, question: str, top_k: int, threshold: float) -> List[DocumentChunk]:
        """Retrieve and filter relevant chunks"""
        # Get query embedding
        query_embedding = self.embedding_service.embedding_model.encode_single(question)
        
        # Search vector store
        chunks = self.embedding_service.vector_store.search(query_embedding, top_k)
        
        # Filter by threshold
        filtered_chunks = [chunk for chunk in chunks if chunk.similarity_score >= threshold]
        
        logger.info(f"Retrieved {len(filtered_chunks)}/{len(chunks)} chunks above threshold {threshold}")
        return filtered_chunks
    
    async def _answer_with_fallback(self, question: str, chunks: List[DocumentChunk]) -> AnswerResult:
        """Try Groq API first, fallback to local model"""
        
        # Try Groq API first if available
        if self.groq_llm and not self.fallback_mode:
            try:
                result = await self.groq_llm.answer_question(question, chunks)
                
                # Check if Groq failed (common errors)
                if (result.confidence == 0.0 or 
                    "error" in result.answer.lower() or 
                    "apologize" in result.answer.lower()):
                    logger.warning("Groq API returned low confidence/error, falling back to local model")
                    raise Exception("Groq API returned poor result")
                
                logger.info(f"Groq API succeeded with confidence: {result.confidence:.3f}")
                return result
                
            except Exception as e:
                logger.warning(f"Groq API failed: {str(e)}, falling back to local model")
                # Set temporary fallback mode to avoid repeated API calls
                self.fallback_mode = True
                
                # Reset fallback mode after 30 seconds
                asyncio.create_task(self._reset_fallback_mode())
        
        # Use local model
        return await self._answer_with_local_model(question, chunks)
    
    async def _answer_with_local_model(self, question: str, chunks: List[DocumentChunk]) -> AnswerResult:
        """Answer using local QA model"""
        start_time = time.time()
        
        # Prepare context from top chunks
        context = self._prepare_context_for_local(chunks)
        
        # Get answer from local model
        local_result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.local_qa.answer_question,
            question,
            context
        )
        
        # Enhanced answer processing for better responses
        enhanced_answer = self._enhance_local_answer(local_result["answer"], question, context)
        
        processing_time = time.time() - start_time
        
        return AnswerResult(
            question=question,
            answer=enhanced_answer,
            confidence=local_result["confidence"],
            reasoning=f"Answer extracted using local AI model from {len(chunks)} relevant document sections.",
            source_chunks=chunks[:3],  # Top 3 sources
            processing_time=processing_time
        )
    
    def _prepare_context_for_local(self, chunks: List[DocumentChunk], max_length: int = 3500) -> str:
        """Prepare context optimized for local QA model"""
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            content = chunk.content.strip()
            
            # Add some structure for better understanding
            section_header = f"\n=== Document Section {i+1} ===\n"
            full_content = section_header + content
            
            if current_length + len(full_content) > max_length:
                break
            
            context_parts.append(full_content)
            current_length += len(full_content)
        
        return "\n".join(context_parts)
    
    def _enhance_local_answer(self, answer: str, question: str, context: str) -> str:
        """Enhance local model answer with better formatting"""
        if not answer or len(answer.strip()) < 10:
            return "The document does not contain sufficient information to answer this question."
        
        # Clean up the answer
        answer = answer.strip()
        
        # If answer seems incomplete, try to extend it
        if len(answer) < 50 and answer.endswith('.'):
            # Look for more relevant information in context
            answer_lower = answer.lower()
            context_sentences = context.split('.')
            
            for sentence in context_sentences:
                if (len(sentence.strip()) > 20 and 
                    any(word in sentence.lower() for word in answer_lower.split()[:3])):
                    extended = sentence.strip()
                    if extended not in answer and len(answer + extended) < 300:
                        answer = answer + " " + extended + "."
                        break
        
        return answer
    
    def _enhance_answer(self, result: AnswerResult, question: str, chunks: List[DocumentChunk], start_time: float) -> AnswerResult:
        """Enhance the final answer result"""
        
        # Calculate enhanced confidence
        enhanced_confidence = self._calculate_enhanced_confidence(
            result.answer, 
            result.confidence, 
            chunks
        )
        
        # Generate better reasoning
        enhanced_reasoning = self._generate_enhanced_reasoning(
            question, 
            result.answer, 
            chunks,
            result.reasoning
        )
        
        return AnswerResult(
            question=question,
            answer=result.answer,
            confidence=enhanced_confidence,
            reasoning=enhanced_reasoning,
            source_chunks=chunks[:3],
            processing_time=time.time() - start_time
        )
    
    def _calculate_enhanced_confidence(self, answer: str, base_confidence: float, chunks: List[DocumentChunk]) -> float:
        """Calculate enhanced confidence score"""
        confidence = base_confidence
        
        # Boost confidence if answer contains specific details
        if any(indicator in answer.lower() for indicator in 
               ['days', 'months', 'years', '%', 'percent', 'limit', 'amount', 'coverage']):
            confidence = min(1.0, confidence + 0.15)
        
        # Boost confidence based on reranking scores
        if chunks and hasattr(chunks[0], 'rerank_score') and chunks[0].rerank_score:
            avg_rerank_score = sum(getattr(chunk, 'rerank_score', 0) for chunk in chunks[:3]) / min(3, len(chunks))
            if avg_rerank_score > 0.7:
                confidence = min(1.0, confidence + 0.1)
        
        # Reduce confidence for uncertain answers
        uncertainty_phrases = [
            'not specified', 'unclear', 'not mentioned', 
            'cannot determine', 'insufficient information', 'document does not'
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence = max(0.1, confidence - 0.3)
        
        return confidence
    
    def _generate_enhanced_reasoning(self, question: str, answer: str, chunks: List[DocumentChunk], base_reasoning: str) -> str:
        """Generate enhanced reasoning"""
        if not chunks:
            return "No relevant context found for this question."
        
        # Build reasoning based on source quality
        reasoning_parts = []
        
        # Source information
        top_chunk = chunks[0]
        if hasattr(top_chunk, 'rerank_score') and top_chunk.rerank_score:
            reasoning_parts.append(f"Answer derived from {len(chunks)} document sections with reranking score: {top_chunk.rerank_score:.3f}")
        else:
            reasoning_parts.append(f"Answer extracted from {len(chunks)} relevant document sections")
        
        # Similarity information
        if hasattr(top_chunk, 'similarity_score') and top_chunk.similarity_score:
            reasoning_parts.append(f"Top similarity score: {top_chunk.similarity_score:.3f}")
        
        # Answer quality assessment
        if len(answer) > 100:
            reasoning_parts.append("Detailed answer found in source material")
        elif len(answer) > 50:
            reasoning_parts.append("Specific information located in document")
        else:
            reasoning_parts.append("Concise answer extracted from context")
        
        return ". ".join(reasoning_parts) + "."
    
    def _no_context_result(self, question: str, start_time: float) -> AnswerResult:
        """Return result when no relevant context found"""
        return AnswerResult(
            question=question,
            answer="I could not find relevant information in the document to answer this question.",
            confidence=0.0,
            reasoning="No document sections met the relevance threshold for this question.",
            source_chunks=[],
            processing_time=time.time() - start_time
        )
    
    async def _reset_fallback_mode(self):
        """Reset fallback mode after delay"""
        await asyncio.sleep(30)
        self.fallback_mode = False
        logger.info("Reset fallback mode - will try Groq API again")
