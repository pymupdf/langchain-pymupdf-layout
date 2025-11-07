from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain_pymupdf_layout

from langchain_pymupdf_layout import PyMuPDFLayoutLoader

loader = PyMuPDFLayoutLoader(
    #file_path="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    file_path="https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",

    # Headers to use for GET request to download a file from a web path
    # (if file_path is a web url)
    ## headers=None,

    # Password for opening encrypted PDF
    ## password=None,
 
    # Additional keyword arguments to pass directly to the
    # underlying `pymupdf4llm.to_markdown` function.
    # See: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html#to_markdown
    # And: https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html#pymupdf-layout-and-parameter-caveats

    # Define which pages of the PDF you want to convert to Markdown.
    # By default, all pages are converted.
    # This would convert the first 2 pages only:
    ## pages = [0,1],

    show_progress=True,
    
)


# Main function
def main():

    print(f"****************")
    print(langchain_pymupdf_layout.version())

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