# langchain-pymupdf-layout

A document loader with [PyMuPDF & PyMuPDF Layout](https://pymupdf.readthedocs.io/en/latest/rag.html#pymupdf-pymupdf-layout) for **LangChain**.

PyMuPDF is a layout-aware document engine for AI and a feature-rich Python toolkit for extracting, rendering and editing PDFs and other documents. Since version 2.0, its ML-based layout analysis turns PDFs into high-quality Markdown for RAG and agents, running fast on CPU and grounding every result to its source page.

Load PDF content to Markdown using AI-based, CPU only, layout analysis.

[![LangChain](https://img.shields.io/badge/LangChain-v1.5.2+-blue)](https://github.com/langchain-ai/langchain)
[![Python version](https://img.shields.io/badge/python-3.11+-blue)](https://pypi.org/project/pymupdf-layout/)
[![License AGPL](https://img.shields.io/github/license/pymupdf/pymupdf)](https://github.com/pymupdf/PyMuPDF/blob/master/COPYING)

## Features

- 📚 Structured data extraction from your documents
- 🧐 Advanced document page layout understanding, including semantic markup for titles, headings, headers, footers, tables, images and text styling
- 🔍 Detect and isolate header and footer patterns on each page

For more detailed information visit the [official PyMuPDF Layout documentation webpage](https://pymupdf.readthedocs.io/en/latest/rag.html#pymupdf-pymupdf-layout).

## Requirements

- Python 3.11 or higher
- LangChain Core v1.5.2 or higher
- PyMuPDF v2.1 or higher

## Installation

Install the package using pip to start using the Document Loader:

```bash
pip install -U langchain-pymupdf-layout

```

## Usage

You can easily integrate and use the PyMuPDF Layout Loader in your Python application for loading and parsing PDFs.

Below is an example of how to set up and utilize this loader.

The example uses `RecursiveCharacterTextSplitter` to chunk the loaded documents. In LangChain v1.0+ the text splitters are a separate package, which is not installed with `langchain`, so install it first, either with the `examples` extra:

```bash
pip install -U "langchain-pymupdf-layout[examples]"
```

or directly:

```bash
pip install -U langchain-text-splitters
```

```python
from langchain_pymupdf_layout import version

print(version())  # Output: version number

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pymupdf_layout import PyMuPDFLayoutLoader

loader = PyMuPDFLayoutLoader(
    file_path="https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
)

documents = loader.load()

# Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} document(s)")
print(f"Created {len(chunks)} chunk(s)")

content = chunks[0].page_content
print(f"\ncontent:\n{content}")

```
