"""
Vision Service Entry Point.

This file serves as the main entry point for the Vision Service.
It configures the Python path and imports the FastAPI app.

Usage:
    python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the FastAPI app from api.main
from api.main import app  # noqa: E402

# Re-export for uvicorn
__all__ = ["app"]
