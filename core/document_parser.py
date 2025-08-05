import PyPDF2
import docx
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os
from abc import ABC, abstractmethod
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

class DocumentParser(ABC):
    """Abstract base class for document parsers"""
    
    @abstractmethod
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse document and extract content"""
        pass

class PDFParser(DocumentParser):
    """Parser for PDF documents using PyMuPDF for better performance"""
    
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF document"""
        try:
            doc = fitz.open(file_path)
            content = []
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "format": "pdf"
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    content.append({
                        "page_number": page_num + 1,
                        "content": text.strip()
                    })
            
            doc.close()
            
            # Clean up temp file if it was downloaded
            if file_path.startswith(tempfile.gettempdir()):
                os.unlink(file_path)
            
            return {
                "content": content,
                "metadata": metadata,
                "total_text": "\n\n".join([page["content"] for page in content])
            }
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise

class DOCXParser(DocumentParser):
    """Parser for DOCX documents"""
    
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse DOCX document"""
        try:
            doc = docx.Document(file_path)
            content = []
            
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    content.append({
                        "paragraph_number": i + 1,
                        "content": paragraph.text.strip()
                    })
            
            metadata = {
                "paragraph_count": len(content),
                "format": "docx"
            }
            
            # Clean up temp file if it was downloaded
            if file_path.startswith(tempfile.gettempdir()):
                os.unlink(file_path)
            
            return {
                "content": content,
                "metadata": metadata,
                "total_text": "\n\n".join([para["content"] for para in content])
            }
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
            raise

class TextParser(DocumentParser):
    """Parser for plain text documents"""
    
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse text document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = [{
                "line_number": i + 1,
                "content": line
            } for i, line in enumerate(lines)]
            
            metadata = {
                "line_count": len(lines),
                "format": "text"
            }
            
            # Clean up temp file if it was downloaded
            if file_path.startswith(tempfile.gettempdir()):
                os.unlink(file_path)
            
            return {
                "content": content,
                "metadata": metadata,
                "total_text": text
            }
            
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {str(e)}")
            raise

class DocumentParserFactory:
    """Factory for creating appropriate document parsers"""
    
    _parsers = {
        '.pdf': PDFParser,
        '.docx': DOCXParser,
        '.doc': DOCXParser,  # Basic support for .doc files
        '.txt': TextParser,
    }
    
    @classmethod
    def create_parser(cls, file_path: str) -> DocumentParser:
        """Create appropriate parser based on file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext not in cls._parsers:
            # Try to infer from content or default to text
            logger.warning(f"Unknown file extension {ext}, defaulting to text parser")
            return TextParser()
        
        return cls._parsers[ext]()
