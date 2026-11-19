"""
Langchain PyMuPDF Layout - Load PDF content to Markdown using AI-based, CPU only, layout analysis
"""

__version__ = "1.0.0"

def version():
    return __version__

from .pymupdf_layout_loader import PyMuPDFLayoutLoader
from .pymupdf_layout_parser import PyMuPDFLayoutParser
 