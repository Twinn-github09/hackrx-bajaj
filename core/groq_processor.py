from typing import List, Dict, Any, Optional
import os
import asyncio
import time
import logging
from groq import AsyncGroq
from models.schemas import DocumentChunk, AnswerResult

logger = logging.getLogger(__name__)

class GroqLLM:
    """Groq-powered LLM for ultra-fast question answering"""
    
    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = None
        self.max_tokens = int(os.getenv("GROQ_MAX_TOKENS", 1000))
        self.temperature = float(os.getenv("GROQ_TEMPERATURE", 0.1))
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
    
    def initialize(self):
        """Initialize the Groq client"""
        if self.client is None:
            self.client = AsyncGroq(api_key=self.api_key)
            logger.info(f"Initialized Groq client with model: {self.model}")
            logger.info(f"API key configured: {self.api_key[:10]}..." if self.api_key else "No API key found")
        else:
            logger.info("Groq client already initialized")
    
    async def answer_question(
        self, 
        question: str, 
        context_chunks: List[DocumentChunk],
        max_context_length: int = 3000
    ) -> AnswerResult:
        """Answer question using Groq API with retrieved context"""
        start_time = time.time()
        
        if not self.client:
            self.initialize()
        
        # Prepare context from chunks
        context = self._prepare_context(context_chunks, max_context_length)
        
        try:
            # Create optimized prompt for insurance/legal domain
            prompt = self._create_prompt(question, context)
            
            logger.info(f"Calling Groq API with model: {self.model}")
            
            # Call Groq API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI assistant specializing in analyzing insurance policies, legal documents, and compliance materials. Provide accurate, concise answers based strictly on the provided context. If information is not in the context, clearly state that."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                stream=False
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            logger.info(f"Groq API response received, length: {len(answer)}")
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer, context_chunks)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(question, answer, context_chunks)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Groq processed question in {processing_time:.3f}s with confidence {confidence:.3f}")
            
            return AnswerResult(
                question=question,
                answer=answer,
                confidence=confidence,
                reasoning=reasoning,
                source_chunks=context_chunks[:3],  # Include top 3 source chunks
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error with Groq API: {str(e)}")
            processing_time = time.time() - start_time
            
            return AnswerResult(
                question=question,
                answer="I apologize, but I encountered an error while processing this question with the AI model.",
                confidence=0.0,
                reasoning=f"Groq API error: {str(e)}",
                source_chunks=[],
                processing_time=processing_time
            )
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create an optimized prompt for insurance/legal domain"""
        prompt = f"""Based on the following document context, please answer the question accurately and concisely.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context
2. Be specific and include relevant details (dates, percentages, conditions)
3. If the exact answer isn't in the context, state "The document does not specify this information"
4. Keep the answer concise but complete.
5. Use the same terminology as in the original document
6. Example query : "Does this policy cover maternity expenses, and what are the conditions?", Expected Answer :"Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
7. Do not include unwanted information in the answer.

ANSWER:"""
        
        return prompt
    
    def _prepare_context(self, chunks: List[DocumentChunk], max_length: int) -> str:
        """Prepare context from chunks, prioritizing most relevant"""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        # Sort chunks by similarity score (if available)
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: x.similarity_score or 0, 
            reverse=True
        )
        
        for i, chunk in enumerate(sorted_chunks):
            content = chunk.content.strip()
            
            # Add chunk number for reference
            relevance_score = f"{chunk.similarity_score:.3f}" if chunk.similarity_score else "N/A"
            chunk_header = f"\n--- Document Section {i+1} (Relevance: {relevance_score}) ---\n"
            full_content = chunk_header + content
            
            if current_length + len(full_content) > max_length:
                # Add as much as we can fit
                remaining = max_length - current_length
                if remaining > 200:  # Only add if meaningful content can fit
                    truncated = full_content[:remaining] + "\n[...truncated...]"
                    context_parts.append(truncated)
                break
            
            context_parts.append(full_content)
            current_length += len(full_content)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, answer: str, chunks: List[DocumentChunk]) -> float:
        """Calculate confidence score based on answer characteristics"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if answer contains specific details
        if any(indicator in answer.lower() for indicator in ['%', 'percent', 'days', 'months', 'years', '$', 'dollar']):
            confidence += 0.2
        
        # Increase confidence if answer is detailed (not too short)
        if len(answer) > 50:
            confidence += 0.1
        
        # Increase confidence based on source chunk relevance
        if chunks and chunks[0].similarity_score:
            avg_similarity = sum(chunk.similarity_score or 0 for chunk in chunks[:3]) / min(3, len(chunks))
            confidence += avg_similarity * 0.2
        
        # Decrease confidence if answer indicates uncertainty
        uncertainty_indicators = ['not specified', 'unclear', 'not mentioned', 'cannot determine']
        if any(indicator in answer.lower() for indicator in uncertainty_indicators):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(
        self, 
        question: str, 
        answer: str, 
        chunks: List[DocumentChunk]
    ) -> str:
        """Generate reasoning for the answer"""
        if not chunks:
            return "No relevant context found for this question."
        
        # Count relevant sources
        high_relevance_count = sum(1 for chunk in chunks if (chunk.similarity_score or 0) > 0.7)
        total_chunks = len(chunks)
        
        reasoning_parts = []
        
        # Source information
        reasoning_parts.append(f"Answer derived from {total_chunks} document sections")
        if high_relevance_count > 0:
            reasoning_parts.append(f"with {high_relevance_count} highly relevant matches")
        
        # Context quality
        if chunks[0].similarity_score and chunks[0].similarity_score > 0.8:
            reasoning_parts.append("High confidence match found in document")
        elif chunks[0].similarity_score and chunks[0].similarity_score > 0.6:
            reasoning_parts.append("Good contextual match identified")
        else:
            reasoning_parts.append("Moderate contextual relevance")
        
        # Processing method
        reasoning_parts.append("Processed using Groq Llama-3.1-8b-instant for fast, accurate analysis")
        
        return ". ".join(reasoning_parts) + "."

class GroqQueryProcessor:
    """Query processor using Groq LLM"""
    
    def __init__(self, embedding_service, groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant"):
        self.embedding_service = embedding_service
        self.groq_llm = GroqLLM(api_key=groq_api_key, model=groq_model)
        
        # Keep reranker for improved retrieval
        from core.llm_processor import RerankerModel
        self.reranker = RerankerModel()
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Groq-powered query processor...")
        self.embedding_service.initialize()
        self.groq_llm.initialize()
        self.reranker.load_model()
        logger.info("Groq query processor initialized successfully")
    
    async def process_query(
        self, 
        question: str,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        similarity_threshold: float = 0.3
    ) -> AnswerResult:
        """Process a single query end-to-end with Groq"""
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
                    answer="I couldn't find relevant information in the document to answer this question.",
                    confidence=0.0,
                    reasoning="No relevant context found above similarity threshold",
                    source_chunks=[],
                    processing_time=time.time() - start_time
                )
            
            # 2. Rerank chunks for better relevance
            reranked_chunks = self.reranker.rerank(question, filtered_chunks, top_k_rerank)
            
            # 3. Generate answer using Groq
            result = await self.groq_llm.answer_question(question, reranked_chunks)
            
            logger.info(f"Processed query with Groq in {result.processing_time:.2f}s, confidence: {result.confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            return AnswerResult(
                question=question,
                answer="I apologize, but I encountered an error while processing this question.",
                confidence=0.0,
                reasoning=f"Processing error: {str(e)}",
                source_chunks=[],
                processing_time=time.time() - start_time
            )
