from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain_pymupdf_layout

from langchain_pymupdf_layout import PyMuPDFLayoutLoader

loader = PyMuPDFLayoutLoader(
    file_path="https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",

    # Password for opening encrypted PDF
    ## password=None,
 
    # Additional keyword arguments to pass directly to the
    # underlying `Document.to_markdown` PyMuPDF function.
    # See: https://pymupdf.readthedocs.io/en/latest/document.html#Document.to_markdown

    # Define which pages of the PDF you want to convert to Markdown.
    # By default, all pages are converted.
    # This would convert the first 2 pages only:
    ## pages = [0,1],
)


# Main function
def main():

    print(f"Using langchain-pymupdf-layout version: ",langchain_pymupdf_layout.version())

    documents = loader.load()
    
    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Loaded {len(documents)} document(s)")
    print(f"Created {len(chunks)} chunk(s)")

    content = chunks[0].page_content
    print(f"\ncontent:\n{content}")
        
    return chunks

if __name__ == "__main__":
    chunks = main()