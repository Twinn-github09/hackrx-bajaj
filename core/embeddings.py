from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from pathlib import Path
import logging
from copy import deepcopy

# Import or define your DocumentChunk dataclass/schema accordingly
from models.schemas import DocumentChunk  

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TextChunker:
    """Intelligent text chunking with overlap and semantic awareness"""
    
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks with better semantic preservation"""
        if not text.strip():
            return []
        
        text = self._clean_text(text)
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=metadata or {},
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                
                overlap_text = self._create_overlap(current_chunk)
                current_chunk = overlap_text + "\n\n" + paragraph
                current_length = len(current_chunk)
                chunk_index += 1
            else:
                current_chunk += ("\n\n" + paragraph) if current_chunk else paragraph
                current_length += paragraph_length
        
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
        import re
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
        text = re.sub(r'\n[ \t]+', '\n', text)   # Remove leading spaces on lines
        return text.strip()
    
    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = [para.strip() for para in paragraphs if para.strip() and len(para.strip()) > 20]
        return cleaned_paragraphs
    
    def _create_overlap(self, chunk: str) -> str:
        sentences = chunk.split('.')
        if len(sentences) <= 2:
            return chunk
        
        overlap_sentences = sentences[-3:]  # Last 3 sentences
        overlap_text = '.'.join(overlap_sentences).strip()
        
        if len(overlap_text) > self.overlap:
            words = overlap_text.split()
            if len(words) > self.overlap // 6:
                overlap_text = ' '.join(words[-(self.overlap // 6):])
        
        return overlap_text


class FastEmbeddingModel:
    """Fast local embedding model using SentenceTransformers"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
    
    def load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with dimension: {self.dimension}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
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
        
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def create_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity after normalization
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        if self.index is None:
            self.create_index()
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[DocumentChunk]:
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = deepcopy(self.chunks[idx])  # To avoid mutating original
                chunk.similarity_score = float(score)
                results.append(chunk)
        
        return results
    
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Saved index with {len(self.chunks)} chunks to {self.index_path}")
    
    def load_index(self) -> bool:
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
        self.index = None
        self.chunks = []
        logger.info("Cleared vector store")

class EmbeddingService:
    """Enhanced embedding service with improved chunking, search, clause retrieval and rationale"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", 
                 index_path: str = "./data/faiss_index"):
        self.embedding_model = FastEmbeddingModel(model_name)
        
        chunk_size = int(os.getenv("MAX_CHUNK_SIZE", 1200))
        overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        
        self.vector_store: FAISSVectorStore = None
        self.index_path = index_path
    
    def initialize(self):
        logger.info("Initializing enhanced embedding service...")
        self.embedding_model.load_model()
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_path=self.index_path
        )
        if not self.vector_store.load_index():
            logger.info("No existing index found, will create new one")
    
    def process_document(self, text: str, metadata: Dict[str, Any] = None) -> int:
        if not self.vector_store:
            self.initialize()
        
        logger.info("Processing document with enhanced algorithm...")
        self.vector_store.clear()
        
        chunks = self.chunker.chunk_text(text, metadata)
        if not chunks:
            logger.warning("No chunks created from document")
            return 0
        
        logger.info(f"Created {len(chunks)} enhanced chunks (size: {self.chunker.chunk_size}, overlap: {self.chunker.overlap})")
        
        chunk_texts = [chunk.content for chunk in chunks]
        batch_size = int(os.getenv("BATCH_SIZE", 32))
        embeddings = self.embedding_model.encode(chunk_texts, batch_size=batch_size)
        
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save_index()
        
        logger.info(f"Successfully processed document into {len(chunks)} chunks with embeddings")
        return len(chunks)
    
    def search_similar(self, query: str, top_k: int = 10, threshold: float = 0.5) -> List[DocumentChunk]:
        if not self.vector_store or len(self.vector_store.chunks) == 0:
            logger.warning("No document indexed for search")
            return []
        
        query_embedding = self.embedding_model.encode_single(query)
        
        search_k = min(top_k * 2, len(self.vector_store.chunks))
        results = self.vector_store.search(query_embedding, search_k)
        
        filtered_results = [chunk for chunk in results if chunk.similarity_score >= threshold]
        final_results = filtered_results[:top_k]
        
        logger.info(f"Enhanced search: {len(final_results)}/{len(results)} chunks above threshold {threshold}")
        if final_results:
            logger.info(f"Top similarity score: {final_results[0].similarity_score:.3f}")
        
        return final_results

    def retrieve_clauses(self, query_clause: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve clauses most similar to a query clause, with explainable rationale.
        """
        results = self.search_similar(query_clause, top_k=top_k, threshold=threshold)
        explainable_results = []
        for rank, chunk in enumerate(results):
            rationale = (
                f"Matched based on semantic similarity score ({chunk.similarity_score:.3f}). "
                f"Top keywords: {', '.join(self.extract_keywords(chunk.content, n=5))}."
            )
            explainable_results.append({
                "rank": rank + 1,
                "similarity_score": chunk.similarity_score,
                "clause_text": chunk.content,
                "rationale": rationale
            })
        return explainable_results

    @staticmethod
    def extract_keywords(text: str, n: int = 5) -> List[str]:
        """
        Basic keyword extractor using token frequency excluding common stopwords.
        Replace or enhance with NLP libraries if needed.
        """
        from collections import Counter
        import re

        tokens = re.findall(r'\b\w+\b', text.lower())
        stopwords = set(["the", "and", "is", "of", "to", "a", "in", "for", "on", "by", "an"])
        keywords = [tok for tok in tokens if tok not in stopwords and len(tok) > 2]
        most_common = [word for word, _ in Counter(keywords).most_common(n)]
        return most_common
