from typing import Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain_pymupdf_layout

from langchain_pymupdf_layout import PyMuPDFLayoutLoader

# PDFs to load: local file paths or URLs can be mixed.
PDF_SOURCES = [
    "https://arxiv.org/pdf/2607.11276",
    "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
]


def lazy_load_all(sources: list[str]) -> Iterator[Document]:
    """Yield documents from each PDF, one file at a time.

    A loader downloads its file (for URLs) as soon as it is created, so each
    loader is created only when the loop reaches that file. Nothing is
    downloaded or parsed until the caller asks for the next document.
    """
    for source in sources:
        loader = PyMuPDFLayoutLoader(
            file_path=source,

            # One Markdown document per PDF (the default is mode="json")
            mode="markdown",

            # Remove header & footer content from pages
            header=False,
            footer=False,
        )
        yield from loader.lazy_load()


# Main function
def main():

    print("Using langchain-pymupdf-layout version:", langchain_pymupdf_layout.version())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Each document is chunked as soon as it is parsed, so only one PDF's
    # full text is held in memory at a time.
    chunks = []
    for i, document in enumerate(lazy_load_all(PDF_SOURCES), start=1):
        doc_chunks = text_splitter.split_documents([document])
        chunks.extend(doc_chunks)

        print(f"\nDocument {i} of {len(PDF_SOURCES)}")
        print(f"  source:     {document.metadata['source']}")
        print(f"  pages:      {document.metadata['total_pages']}")
        print(f"  characters: {len(document.page_content)}")
        print(f"  chunks:     {len(doc_chunks)}")

    print(f"\nCreated {len(chunks)} chunk(s) in total from {len(PDF_SOURCES)} PDF(s)")

    return chunks


if __name__ == "__main__":
    chunks = main()
