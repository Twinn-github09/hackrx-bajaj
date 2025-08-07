import PyPDF2
import docx
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os
from abc import ABC, abstractmethod
import fitz  # PyMuPDFac    
import logging
import email
from email import policy
from email.parser import BytesParser 
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
class EMLParser(DocumentParser):
    """Parser for EML email documents"""

    async def parse(self, file_path: str) -> Dict[str, Any]:
        """Parse EML document"""
        try:
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)

            subject = msg.get("subject", "")
            sender = msg.get("from", "")
            recipient = msg.get("to", "")
            date = msg.get("date", "")
            body = self._get_body(msg)

            metadata = {
                "subject": subject,
                "from": sender,
                "to": recipient,
                "date": date,
                "format": "eml"
            }

            content = [{"part": "body", "content": body.strip()}] if body.strip() else []

            # Clean up temp file if it was downloaded
            if file_path.startswith(tempfile.gettempdir()):
                os.unlink(file_path)

            return {
                "content": content,
                "metadata": metadata,
                "total_text": body.strip()
            }

        except Exception as e:
            logger.error(f"Error parsing EML file {file_path}: {str(e)}")
            raise

    def _get_body(self, msg) -> str:
        """Extract plain text body from email"""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    return part.get_content()
        else:
            return msg.get_content()
        return ""
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
        '.doc': DOCXParser,
        '.txt': TextParser,
        '.eml': EMLParser  
    }

    @classmethod
    def create_parser(cls, file_path: str) -> DocumentParser:
        ext = Path(file_path).suffix.lower()

        if ext not in cls._parsers:
            logger.warning(f"Unknown file extension {ext}, defaulting to text parser")
            return TextParser()

        return cls._parsers[ext]()

