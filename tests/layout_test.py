import json
from collections import Counter

import pymupdf
import langchain_pymupdf_layout

from langchain_pymupdf_layout import PyMuPDFLayoutLoader

PDF_SOURCE = "https://arxiv.org/pdf/2607.11276"


def box_mode():
    """One LangChain document per layout box: heading, paragraph, table, ..."""
    print("\n=== mode='json', layout_chunking='box' ===")

    loader = PyMuPDFLayoutLoader(
        file_path=PDF_SOURCE,
        mode="json",
        layout_chunking="box",

        # Remove header & footer content from pages
        header=False,
        footer=False,
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")
    print("By category:", dict(Counter(d.metadata["category"] for d in documents)))

    # Every document knows its page and position, in PDF points.
    for document in documents[:5]:
        m = document.metadata
        print(
            f"  page {m['page']} {m['category']:<15}"
            f" ({m['x0']:.0f}, {m['y0']:.0f}, {m['x1']:.0f}, {m['y1']:.0f})"
            f"  {document.page_content[:60]!r}"
        )

    # Tables come through whole, as Markdown tables.
    table = next((d for d in documents if d.metadata["category"] == "table"), None)
    if table:
        print(f"\nFirst table (page {table.metadata['page']}):")
        print(table.page_content)

        # Use the position to highlight the table in the original PDF.
        # The loader keeps the downloaded PDF at loader.file_path.
        with pymupdf.open(loader.file_path) as pdf:
            page = pdf[table.metadata["page"]]
            rect = pymupdf.Rect(*(table.metadata[k] for k in ("x0", "y0", "x1", "y1")))
            page.draw_rect(rect, color=(1, 0, 0), width=2)
            page.get_pixmap(dpi=100).save("layout_test_table.png")
        print("\nSaved the page with the table outlined in red to layout_test_table.png")

    return documents


def section_mode():
    """One LangChain document per section: a heading plus everything under it."""
    print("\n=== mode='json', layout_chunking='section' ===")

    loader = PyMuPDFLayoutLoader(
        file_path=PDF_SOURCE,
        mode="json",
        layout_chunking="section",
        header=False,
        footer=False,
    )

    # lazy_load() hands sections over one at a time.
    sections = []
    for document in loader.lazy_load():
        sections.append(document)
        m = document.metadata
        pages = f"{m['page']}" if m["page"] == m["page_end"] else f"{m['page']}-{m['page_end']}"
        print(
            f"  pages {pages:<5} {len(document.page_content):>5} chars"
            f"  {m['heading'][:60] or '(before first heading)'}"
        )

    print(f"Loaded {len(sections)} section(s)")

    # `regions` lists the page and position of every box in the section.
    first = sections[1] if len(sections) > 1 else sections[0]
    regions = json.loads(first.metadata["regions"])
    print(f"\nSection {first.metadata['heading']!r} is made of {len(regions)} box(es):")
    for region in regions[:5]:
        print(f"  page {region['page']} {region['category']:<15} bbox {region['bbox']}")

    print(f"\nSection text (Markdown):\n{first.page_content[:800]}")

    return sections


# Main function
def main():

    print("Using langchain-pymupdf-layout version:", langchain_pymupdf_layout.version())

    box_documents = box_mode()
    section_documents = section_mode()

    return box_documents, section_documents


if __name__ == "__main__":
    main()
