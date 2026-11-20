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
- 📍 One LangChain document per layout element or per section, each with its page and position in the PDF, or optionally the whole PDF as one Markdown document

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

The loader uses PyMuPDF Layout to find the headings, paragraphs, lists, tables and other elements on each page.

By default (`mode="json"`) it uses this layout analysis, from `Document.to_json`, to return many small LangChain documents: one per element, or one per section. Each document records where its content is in the PDF, so you can:

- show or highlight the exact source of an answer
- filter by element type, for example index only body text and tables
- keep tables whole, as Markdown tables, rather than letting a text splitter cut through them

Use `layout_chunking` to choose how the content is grouped.

If you'd rather have one document containing the whole PDF as Markdown, use `mode="markdown"`. See [Markdown mode](#markdown-mode).

### One document per layout element

`layout_chunking="box"` (the default) returns one document per heading, paragraph, list item, table, caption and so on. Pictures that contain no text are skipped.

```python
from langchain_pymupdf_layout import PyMuPDFLayoutLoader

loader = PyMuPDFLayoutLoader(
    file_path="https://arxiv.org/pdf/2607.11276",
    layout_chunking="box",
)

for document in loader.lazy_load():
    m = document.metadata
    print(m["page"], m["category"], (m["x0"], m["y0"], m["x1"], m["y1"]))
    print(document.page_content)
```

Each document's metadata includes:

| Key | Description |
| --- | --- |
| `page` | Page number, starting from 0 |
| `category` | The element type: `title`, `section-header`, `text`, `list-item`, `table`, `picture`, `caption`, `formula`, `footnote`, `page-header` or `page-footer` |
| `header_level` | The heading level for headings, otherwise `0` |
| `x0`, `y0`, `x1`, `y1` | The element's position on the page, in PDF points, with the origin at the top-left corner |
| `page_width`, `page_height` | The page size, in PDF points |

### One document per section

`layout_chunking="section"` returns one document per section: each heading together with everything under it, up to the next heading. The content is written as Markdown. This usually gives better chunks for retrieval than single elements, which can be very small (a heading on its own, for example).

```python
import json

loader = PyMuPDFLayoutLoader(
    file_path="https://arxiv.org/pdf/2607.11276",
    layout_chunking="section",
)

for document in loader.lazy_load():
    m = document.metadata
    print(f"{m['heading']} (pages {m['page']}-{m['page_end']})")

    # The position of every element in the section
    for region in json.loads(m["regions"]):
        print(region["page"], region["category"], region["bbox"])
```

Each document's metadata includes:

| Key | Description |
| --- | --- |
| `page`, `page_end` | The first and last page of the section, starting from 0 |
| `category` | Always `section` |
| `heading` | The section's heading text. Empty for any content before the first heading |
| `header_level` | The level of the section's heading |
| `regions` | A JSON string listing the `page`, `category` and `bbox` (`[x0, y0, x1, y1]`) of every element in the section |

`regions` is stored as a JSON string, rather than a list, because some vector stores only accept plain text and number values in metadata.

### One Markdown document

With `mode="markdown"` the loader returns one document per PDF, containing the whole PDF as Markdown. You can then split it into chunks yourself.

The example below uses `RecursiveCharacterTextSplitter` to chunk the document. In LangChain v1.0+ the text splitters are a separate package, which is not installed with `langchain`, so install it first, either with the `examples` extra:

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
    file_path="https://arxiv.org/pdf/2607.11276",
    mode="markdown",
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

In Markdown mode, any other keyword arguments are passed to `Document.to_markdown`. For example, `header=False` and `footer=False` leave out page headers and footers, and `pages=[0, 1]` converts only the first two pages.

## Notes

- In both modes, every document also has the PDF's own metadata, such as `source`, `total_pages` and `creationdate`.
- `header=False` and `footer=False` work in JSON mode too: they leave out page headers and footers.
- Other keyword arguments are passed to `Document.to_json` (for example `pages`, `use_ocr`, `force_ocr` and `ocr_language`). Options that only apply to Markdown output are ignored in JSON mode.
- See [tests/layout_test.py](tests/layout_test.py) for a full example, including drawing an element's position onto the PDF page.

