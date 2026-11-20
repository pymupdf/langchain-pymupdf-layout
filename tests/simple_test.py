import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain_pymupdf_layout

from langchain_pymupdf_layout import PyMuPDFLayoutLoader

loader = PyMuPDFLayoutLoader(
    file_path="https://arxiv.org/pdf/2607.11276",

    # One Markdown document for the whole PDF (the default is mode="json")
    mode="markdown",

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

    documents = loader.load()

    print(documents[0].page_content[:100]) # print some Markdown content
    
    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    print(f"\nLoaded {len(documents)} document(s)")
    print(f"\nCreated {len(chunks)} chunk(s)")
        
    return chunks

if __name__ == "__main__":
    chunks = main()