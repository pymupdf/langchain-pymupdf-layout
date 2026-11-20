from langchain_pymupdf_layout import PyMuPDFLayoutLoader

PDF_SOURCE = "https://arxiv.org/pdf/2607.11276"

loader = PyMuPDFLayoutLoader(
    file_path=PDF_SOURCE,
)
documents = loader.load()
document = documents[0]

m = document.metadata
print(f"Total pages: {m['total_pages']}")
print(
    f"  page {m['page']} {m['category']:<15}"
    f" ({m['x0']:.0f}, {m['y0']:.0f}, {m['x1']:.0f}, {m['y1']:.0f})"
    f"  {document.page_content[:60]!r}"
)