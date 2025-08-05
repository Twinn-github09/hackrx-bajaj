from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
import aiofiles
from pathlib import Path
import tempfile
import os
from loguru import logger

class DocumentLoader(ABC):
    """Abstract base class for document loaders"""
    
    @abstractmethod
    async def load(self, source: str) -> str:
        """Load document content from source"""
        pass

class URLDocumentLoader(DocumentLoader):
    """Loader for documents from URLs"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    async def load(self, url: str) -> str:
        """Download and return document content from URL"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Create temp file
                suffix = self._get_file_extension(url)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(response.content)
                    temp_path = tmp_file.name
                
                logger.info(f"Downloaded document to {temp_path}")
                return temp_path
                
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {str(e)}")
            raise
    
    def _get_file_extension(self, url: str) -> str:
        """Extract file extension from URL"""
        path = Path(url.split('?')[0])  # Remove query parameters
        return path.suffix or '.pdf'

class LocalDocumentLoader(DocumentLoader):
    """Loader for local documents"""
    
    async def load(self, file_path: str) -> str:
        """Return local file path"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        return str(path.absolute())

class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders"""
    
    @staticmethod
    def create_loader(source: str) -> DocumentLoader:
        """Create appropriate loader based on source type"""
        if source.startswith(('http://', 'https://')):
            return URLDocumentLoader()
        else:
            return LocalDocumentLoader()
