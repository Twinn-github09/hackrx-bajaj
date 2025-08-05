from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from pathlib import Path
import logging
from models.schemas import DocumentChunk
import uuid

logger = logging.getLogger(__name__)

class TextChunker:
    """Intelligent text chunking with overlap and semantic awareness"""
    
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks with better semantic preservation"""
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split by paragraphs first, then sentences for better coherence
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=metadata or {},
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                
                # Create overlap for next chunk
                overlap_text = self._create_overlap(current_chunk)
                current_chunk = overlap_text + "\n\n" + paragraph
                current_length = len(current_chunk)
                chunk_index += 1
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_length += paragraph_length
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata or {},
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving structure"""
        import re
        
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
        text = re.sub(r'\n[ \t]+', '\n', text)   # Remove leading spaces on lines
        
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, keeping related content together"""
        # Split on double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # Only keep substantial paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _create_overlap(self, chunk: str) -> str:
        """Create overlap text from the end of current chunk, preserving sentences"""
        sentences = chunk.split('.')
        if len(sentences) <= 2:
            return chunk
        
        # Take last few sentences for overlap
        overlap_sentences = sentences[-3:]  # Last 3 sentences
        overlap_text = '.'.join(overlap_sentences).strip()
        
        # Ensure overlap doesn't exceed limit
        if len(overlap_text) > self.overlap:
            words = overlap_text.split()
            if len(words) > self.overlap // 6:  # Rough word count limit
                overlap_text = ' '.join(words[-(self.overlap // 6):])
        
        return overlap_text

class FastEmbeddingModel:
    """Fast local embedding model using SentenceTransformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
    
    def load_model(self):
        """Load the embedding model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Get dimension from model
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with dimension: {self.dimension}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.model is None:
            self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.encode([text])[0]

class FAISSVectorStore:
    """FAISS-based vector store for fast similarity search"""
    
    def __init__(self, dimension: int, index_path: str = "./data/faiss_index"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.index_file = self.index_path / "index.faiss"
        self.chunks_file = self.index_path / "chunks.pkl"
        
        # Create directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def create_index(self):
        """Create a new FAISS index"""
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the index"""
        if self.index is None:
            self.create_index()
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store chunks
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[DocumentChunk]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Return chunks with similarity scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx].copy()
                chunk.similarity_score = float(score)
                results.append(chunk)
        
        return results
    
    def save_index(self):
        """Save index and chunks to disk"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"Saved index with {len(self.chunks)} chunks to {self.index_path}")
    
    def load_index(self) -> bool:
        """Load index and chunks from disk"""
        try:
            if self.index_file.exists() and self.chunks_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                
                with open(self.chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                logger.info(f"Loaded index with {len(self.chunks)} chunks from {self.index_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
        
        return False
    
    def clear(self):
        """Clear the index and chunks"""
        self.index = None
        self.chunks = []
        logger.info("Cleared vector store")

class EmbeddingService:
    """Enhanced embedding service with improved chunking and search"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 index_path: str = "./data/faiss_index"):
        self.embedding_model = FastEmbeddingModel(model_name)
        
        # Use improved chunker with environment variables
        chunk_size = int(os.getenv("MAX_CHUNK_SIZE", 1200))
        overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        
        self.vector_store = None
        self.index_path = index_path
    
    def initialize(self):
        """Initialize the embedding service"""
        logger.info("Initializing enhanced embedding service...")
        self.embedding_model.load_model()
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_path=self.index_path
        )
        
        # Try to load existing index
        if not self.vector_store.load_index():
            logger.info("No existing index found, will create new one")
    
    def process_document(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """Process document with enhanced chunking and embedding"""
        if not self.vector_store:
            self.initialize()
        
        logger.info("Processing document with enhanced algorithm...")
        
        # Clear existing index for new document
        self.vector_store.clear()
        
        # Enhanced chunking
        chunks = self.chunker.chunk_text(text, metadata)
        
        if not chunks:
            logger.warning("No chunks created from document")
            return 0
        
        logger.info(f"Created {len(chunks)} enhanced chunks (size: {self.chunker.chunk_size}, overlap: {self.chunker.overlap})")
        
        # Create embeddings with batching
        chunk_texts = [chunk.content for chunk in chunks]
        batch_size = int(os.getenv("BATCH_SIZE", 32))
        embeddings = self.embedding_model.encode(chunk_texts, batch_size=batch_size)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Save index
        self.vector_store.save_index()
        
        logger.info(f"Successfully processed document into {len(chunks)} chunks with embeddings")
        return len(chunks)
    
    def search_similar(self, query: str, top_k: int = 10, threshold: float = 0.5) -> List[DocumentChunk]:
        """Enhanced search with threshold filtering"""
        if not self.vector_store or len(self.vector_store.chunks) == 0:
            logger.warning("No document indexed for search")
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search with more candidates initially
        search_k = min(top_k * 2, len(self.vector_store.chunks))
        results = self.vector_store.search(query_embedding, search_k)
        
        # Filter by threshold
        filtered_results = [chunk for chunk in results if chunk.similarity_score >= threshold]
        
        # Return top_k from filtered results
        final_results = filtered_results[:top_k]
        
        logger.info(f"Enhanced search: {len(final_results)}/{len(results)} chunks above threshold {threshold}")
        
        if final_results:
            logger.info(f"Top similarity score: {final_results[0].similarity_score:.3f}")
        
        return final_results
