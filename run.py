#!/usr/bin/env python3
"""
LLM-Powered Intelligent Query-Retrieval System
Fast RAG implementation with local models

Usage:
    python run.py [--host HOST] [--port PORT] [--reload]
    
Example:
    python run.py --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("🚀 Starting LLM-Powered Intelligent Query-Retrieval System")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
