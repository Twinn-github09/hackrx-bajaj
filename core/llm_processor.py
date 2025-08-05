from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
import logging
from models.schemas import DocumentChunk, AnswerResult
import time
import re

logger = logging.getLogger(__name__)

class FastLocalLLM:
    """Fast local LLM for question answering"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the LLM model"""
        if self.model is None:
            logger.info(f"Loading LLM model: {self.model_name}")
            
            try:
                # For better performance, use a lightweight Q&A model
                qa_model_name = "deepset/roberta-base-squad2"  # Fast and accurate for QA
                
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=qa_model_name,
                    tokenizer=qa_model_name,
                    device=0 if self.device == "cuda" else -1
                )
                
                logger.info(f"Loaded Q&A model on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                # Fallback to a simpler approach
                self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model for text generation"""
        try:
            fallback_model = "distilgpt2"  # Very fast and lightweight
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            # Add padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded fallback model: {fallback_model}")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {str(e)}")
            raise
    
    def answer_question(
        self, 
        question: str, 
        context_chunks: List[DocumentChunk],
        max_context_length: int = 2000
    ) -> AnswerResult:
        """Answer question using retrieved context"""
        start_time = time.time()
        
        if not self.qa_pipeline and not self.model:
            self.load_model()
        
        # Prepare context
        context = self._prepare_context(context_chunks, max_context_length)
        
        try:
            if self.qa_pipeline:
                # Use the Q&A pipeline (preferred)
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=200,
                    handle_impossible_answer=True
                )
                
                answer = result['answer']
                confidence = result['score']
                
            else:
                # Use fallback text generation
                answer, confidence = self._generate_answer_fallback(question, context)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(question, answer, context_chunks)
            
            processing_time = time.time() - start_time
            
            return AnswerResult(
                question=question,
                answer=answer,
                confidence=confidence,
                reasoning=reasoning,
                source_chunks=context_chunks[:3],  # Include top 3 source chunks
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return AnswerResult(
                question=question,
                answer="I apologize, but I encountered an error while processing this question.",
                confidence=0.0,
                reasoning="Error occurred during processing",
                source_chunks=[],
                processing_time=time.time() - start_time
            )
    
    def _prepare_context(self, chunks: List[DocumentChunk], max_length: int) -> str:
        """Prepare context from chunks"""
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            content = chunk.content
            if current_length + len(content) > max_length:
                # Truncate to fit
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful content can fit
                    context_parts.append(content[:remaining] + "...")
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer_fallback(self, question: str, context: str) -> tuple[str, float]:
        """Generate answer using fallback text generation model"""
        try:
            # Create a prompt for answer generation
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            answer_start = generated_text.find("Answer:") + len("Answer:")
            answer = generated_text[answer_start:].strip()
            
            # Clean up answer
            answer = self._clean_answer(answer)
            
            return answer, 0.7  # Default confidence for fallback
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {str(e)}")
            return "I couldn't generate an answer based on the provided context.", 0.0
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the answer"""
        # Remove redundant text and clean up
        answer = re.sub(r'\n+', ' ', answer)  # Replace newlines with spaces
        answer = re.sub(r'\s+', ' ', answer)  # Remove extra spaces
        
        # Stop at first sentence ending if answer is too long
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) > 1 and len(answer) > 200:
            return sentences[0].strip() + "."
        
        return answer.strip()
    
    def _generate_reasoning(
        self, 
        question: str, 
        answer: str, 
        chunks: List[DocumentChunk]
    ) -> str:
        """Generate reasoning for the answer"""
        if not chunks:
            return "No relevant context found for this question."
        
        # Simple reasoning based on chunk sources
        source_info = []
        for i, chunk in enumerate(chunks[:3]):  # Top 3 chunks
            similarity = chunk.similarity_score or 0.0
            source_info.append(f"Source {i+1} (similarity: {similarity:.3f})")
        
        reasoning = f"Answer based on {len(chunks)} relevant document sections. "
        reasoning += f"Primary sources: {', '.join(source_info)}. "
        reasoning += f"The answer was extracted from the most relevant context matching the question about '{question[:50]}...'."
        
        return reasoning

class RerankerModel:
    """Fast reranking model for improving retrieval results"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """Load reranker model"""
        if self.model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded")
    
    def rerank(
        self, 
        query: str, 
        chunks: List[DocumentChunk], 
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """Rerank chunks based on query relevance"""
        if not chunks:
            return []
            
        if not self.model:
            self.load_model()
        
        try:
            # Prepare query-document pairs
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Combine chunks with scores and sort
            chunk_scores = list(zip(chunks, scores))
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update similarity scores and return top_k
            reranked_chunks = []
            for chunk, score in chunk_scores[:top_k]:
                chunk_copy = chunk.copy()
                chunk_copy.similarity_score = float(score)
                reranked_chunks.append(chunk_copy)
            
            logger.info(f"Reranked {len(chunks)} chunks, returning top {len(reranked_chunks)}")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original chunks if reranking fails
            return chunks[:top_k]

class QueryProcessor:
    """Main processor for handling queries with retrieval and generation"""
    
    def __init__(self, embedding_service, llm_model_name: str = None, reranker_model_name: str = None):
        self.embedding_service = embedding_service
        self.llm = FastLocalLLM(llm_model_name) if llm_model_name else FastLocalLLM()
        self.reranker = RerankerModel(reranker_model_name) if reranker_model_name else RerankerModel()
        
    def initialize(self):
        """Initialize all models"""
        logger.info("Initializing query processor...")
        self.embedding_service.initialize()
        self.llm.load_model()
        self.reranker.load_model()
        logger.info("Query processor initialized")
    
    def process_query(
        self, 
        question: str,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        similarity_threshold: float = 0.3
    ) -> AnswerResult:
        """Process a single query end-to-end"""
        start_time = time.time()
        
        try:
            # 1. Retrieve relevant chunks
            retrieved_chunks = self.embedding_service.search_similar(question, top_k_retrieval)
            
            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in retrieved_chunks 
                if (chunk.similarity_score or 0) >= similarity_threshold
            ]
            
            if not filtered_chunks:
                return AnswerResult(
                    question=question,
                    answer="I couldn't find relevant information to answer this question.",
                    confidence=0.0,
                    reasoning="No relevant context found above similarity threshold",
                    source_chunks=[],
                    processing_time=time.time() - start_time
                )
            
            # 2. Rerank chunks
            reranked_chunks = self.reranker.rerank(question, filtered_chunks, top_k_rerank)
            
            # 3. Generate answer
            result = self.llm.answer_question(question, reranked_chunks)
            
            logger.info(f"Processed query in {result.processing_time:.2f}s with {len(reranked_chunks)} context chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            return AnswerResult(
                question=question,
                answer="I apologize, but I encountered an error while processing this question.",
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                source_chunks=[],
                processing_time=time.time() - start_time
            )
