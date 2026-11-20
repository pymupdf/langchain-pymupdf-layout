import json
import logging
import threading

try:
    import pymupdf
except ImportError:
    raise ImportError(
        "pymupdf package not found, please install it "
        "with `pip install pymupdf`"
    )

from datetime import datetime

from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
)

from langchain_core.documents import Document
from langchain_core.document_loaders import (
    Blob,
    BaseBlobParser
)

_STD_METADATA_KEYS = {"source", "total_pages", "creationdate", "creator", "producer"}

# Layout box classes that start a new section in `layout_chunking="section"`.
_HEADING_CLASSES = {"title", "section-header"}

logger = logging.getLogger(__name__)


def _line_text(spans: list[dict[str, Any]]) -> str:
    """Join the spans of one text line.

    PDFs often store each word as its own span with no space character
    between them (common in justified text), so a space is added wherever
    there is a visible horizontal gap between two spans.
    """
    text = ""
    previous = None
    for span in spans:
        span_text = span["text"]
        if previous is not None and text and not text[-1].isspace() and not span_text[:1].isspace():
            gap = span["bbox"][0] - previous["bbox"][2]
            if gap > 0.1 * max(span["size"], previous["size"]):
                text += " "
        text += span_text
        previous = span
    return text


def _box_text(box: dict[str, Any]) -> str:
    """Get the text of one layout box from `Document.to_json` output.

    Tables are returned as Markdown tables. All other boxes are returned as a
    single line of text with whitespace normalized.
    """
    table = box.get("table")
    if table and table.get("markdown"):
        return table["markdown"].strip()
    text = ""
    for line in box.get("textlines") or []:
        line_text = " ".join(_line_text(line.get("spans", [])).split())
        if not line_text:
            continue
        # Keep a word hyphenated across a line break together ("multi-agent").
        if text and not text.endswith("-"):
            text += " "
        text += line_text
    return text


def _rounded_bbox(box: dict[str, Any]) -> list[float]:
    return [round(box[k], 2) for k in ("x0", "y0", "x1", "y1")]

def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate that the metadata has all the standard keys and the page is an integer.

    The standard keys are:
    - source
    - total_page
    - creationdate
    - creator
    - producer

    Validate that page is an integer if it is present.
    """
    if not _STD_METADATA_KEYS.issubset(metadata.keys()):
        raise ValueError("The PDF parser must valorize the standard metadata.")
    if not isinstance(metadata.get("page", 0), int):
        raise ValueError("The PDF metadata page must be a integer.")
    return metadata


def _purge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Purge metadata from unwanted keys and normalize key names.

    Args:
        metadata: The original metadata dictionary.

    Returns:
        The cleaned and normalized the key format of metadata dictionary.
    """
    new_metadata: dict[str, Any] = {}
    map_key = {
        "page_count": "total_pages",
        "file_path": "source",
    }
    for k, v in metadata.items():
        if type(v) not in [str, int]:
            v = str(v)
        if k.startswith("/"):
            k = k[1:]
        k = k.lower()
        if k in ["creationdate", "moddate"]:
            try:
                new_metadata[k] = datetime.strptime(
                    v.replace("'", ""), "D:%Y%m%d%H%M%S%z"
                ).isoformat("T")
            except ValueError:
                new_metadata[k] = v
        elif k in map_key:
            # Normalize key with others PDF parser
            new_metadata[map_key[k]] = v
            new_metadata[k] = v
        elif isinstance(v, str):
            new_metadata[k] = v.strip()
        elif isinstance(v, int):
            new_metadata[k] = v
    return new_metadata


class PyMuPDFLayoutParser(BaseBlobParser):
    """Parse a blob from a PDF using `PyMuPDF

    Examples:
        Setup:

        .. code-block:: bash

            pip install -U langchain-pymupdf-layout

        Load a blob from a PDF file:

        .. code-block:: python

            from langchain_core.documents.base import Blob

            blob = Blob.from_path("./test.pdf")

        Instantiate the parser:

        .. code-block:: python

            from langchain_pymupdf_layout import PyMuPDFLayoutParser

            parser = PyMuPDFLayoutParser()

        Lazily parse the blob:

        .. code-block:: python

            docs = []
            docs_lazy = parser.lazy_parse(blob)

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)
    """

    # PyMuPDF is not thread safe.
    # See https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        password: Optional[str] = None,
        mode: Literal["json", "markdown"] = "json",
        layout_chunking: Literal["box", "section"] = "box",
        **pymupdf_layout_kwargs,
    ) -> None:
        """Initialize a parser to extract PDF content using PyMuPDF Layout.

        Args:
            password: Optional password for opening encrypted PDFs.
            mode: How to turn the PDF into LangChain documents.

                - ``"json"`` (default): several documents made from the layout
                  analysis in `Document.to_json`. Each one records the page and
                  position of its content. See `layout_chunking`.
                - ``"markdown"``: one document for the whole PDF, made with
                  `Document.to_markdown`.
            layout_chunking: How to group layout boxes when `mode="json"`.

                - ``"box"`` (default): one document per layout box (heading,
                  paragraph, list item, table, caption, ...). Metadata includes
                  the box class (`category`), `page` and its position
                  (`x0`, `y0`, `x1`, `y1`, in PDF points).
                - ``"section"``: one document per section: each heading plus
                  the boxes under it, up to the next heading, rendered as
                  Markdown. Metadata includes the `heading`, the first and last
                  page, and the position of every box as a JSON string
                  (`regions`).
            **pymupdf_layout_kwargs: Additional keyword arguments to pass directly to
                `Document.to_json` (json mode) or `Document.to_markdown`
                (markdown mode). In json mode, `header=False` and `footer=False`
                drop page headers and footers.

        Returns:
            This method does not directly return data. Use the `parse` or `lazy_parse`
            methods to retrieve parsed documents with content and metadata.

        Raises:
            ValueError: If `mode` or `layout_chunking` is not a supported value, or
                if unsupported `pymupdf_layout_kwargs` are provided (e.g.,
                `table_strategy`, `detect_bg_color`, `ignore_images`).
        """

        if mode not in ("markdown", "json"):
            raise ValueError(f"mode must be 'markdown' or 'json', not {mode!r}")

        if layout_chunking not in ("box", "section"):
            raise ValueError(
                f"layout_chunking must be 'box' or 'section', not {layout_chunking!r}"
            )

        if "table_strategy" in pymupdf_layout_kwargs:
            raise ValueError("PyMuPDF Layout argument: table_strategy should not be set")
        
        if "ignore_images" in pymupdf_layout_kwargs:
            raise ValueError("PyMuPDF Layout argument: ignore_images should not be set")
        
        if "graphics_limit" in pymupdf_layout_kwargs:
            raise ValueError("PyMuPDF Layout argument: graphics_limit should not be set")
        
        if "detect_bg_color" in pymupdf_layout_kwargs:
            raise ValueError("PyMuPDF Layout argument: detect_bg_color should not be set")
            
        super().__init__()

        self.password = password
        self.mode = mode
        self.layout_chunking = layout_chunking
        self.pymupdf_layout_kwargs = pymupdf_layout_kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse a blob from a PDF document.

        Args:
            blob: The blob from a PDF document to parse.

        Raises:
            ImportError: If the `pymupdf` package is not found.

        Yield:
            An iterator over the parsed documents with PDF content.
        """
        
        with PyMuPDFLayoutParser._lock:
            with blob.as_bytes_io() as file_path:
                if blob.data is None:
                    doc = pymupdf.open(file_path)
                else:
                    doc = pymupdf.open(stream=file_path, filetype="pdf")
                try:
                    if doc.needs_pass and not doc.authenticate(self.password or ""):
                        raise ValueError(
                            "The PDF is password protected and the password is "
                            "missing or incorrect."
                        )
                    metadata = self._extract_metadata(doc, blob)
                    if self.mode == "json":
                        documents = self._get_layout_documents(doc, metadata)
                    else:
                        documents = [
                            Document(
                                page_content=self._get_md(doc),
                                metadata=_validate_metadata(metadata),
                            )
                        ]
                finally:
                    doc.close()

        # Yield outside the lock so a caller that stops iterating early
        # doesn't leave the lock held.
        yield from documents

    def _get_layout_documents(
        self, doc: pymupdf.Document, base_metadata: dict[str, Any]
    ) -> list[Document]:
        """Make LangChain documents from the `Document.to_json` layout analysis.

        Args:
            doc: The PyMuPDF document object.
            base_metadata: Document-level metadata, copied into every document.

        Returns:
            list[Document]: One document per layout box or per section,
                depending on `layout_chunking`.
        """
        params = dict(self.pymupdf_layout_kwargs)
        # `to_json` ignores `header` and `footer`, so apply them here by
        # dropping page header and footer boxes.
        skip_classes = set()
        if not params.pop("header", True):
            skip_classes.add("page-header")
        if not params.pop("footer", True):
            skip_classes.add("page-footer")

        layout = json.loads(doc.to_json(**params))

        # (page index, page dict, box dict, box text) in reading order.
        boxes = []
        for page in layout["pages"]:
            for box in page["boxes"]:
                if box["boxclass"] in skip_classes:
                    continue
                text = _box_text(box)
                if text:  # e.g. pictures with no text in them
                    boxes.append((page["page_number"] - 1, page, box, text))

        if self.layout_chunking == "section":
            return self._section_documents(boxes, base_metadata)

        return [
            Document(
                page_content=text,
                metadata=_validate_metadata(
                    {
                        **base_metadata,
                        "page": page_index,
                        "category": box["boxclass"],
                        "header_level": box.get("header_level") or 0,
                        **dict(zip(("x0", "y0", "x1", "y1"), _rounded_bbox(box))),
                        "page_width": page["width"],
                        "page_height": page["height"],
                    }
                ),
            )
            for page_index, page, box, text in boxes
        ]

    @staticmethod
    def _section_documents(
        boxes: list[tuple[int, dict, dict, str]], base_metadata: dict[str, Any]
    ) -> list[Document]:
        """Group layout boxes into one document per section.

        A section starts at each title or section header and runs until the
        next one. Any content before the first heading becomes a section
        with an empty `heading`.
        """
        sections: list[list[tuple[int, dict, dict, str]]] = []
        for item in boxes:
            if not sections or item[2]["boxclass"] in _HEADING_CLASSES:
                sections.append([])
            sections[-1].append(item)

        documents = []
        for section in sections:
            first_box = section[0][2]
            is_heading = first_box["boxclass"] in _HEADING_CLASSES
            header_level = (first_box.get("header_level") or 1) if is_heading else 0

            parts = []
            for _, _, box, text in section:
                if box["boxclass"] in _HEADING_CLASSES:
                    level = min(box.get("header_level") or 1, 6)
                    text = "#" * level + " " + text
                parts.append(text)

            documents.append(
                Document(
                    page_content="\n\n".join(parts),
                    metadata=_validate_metadata(
                        {
                            **base_metadata,
                            "page": section[0][0],
                            "page_end": section[-1][0],
                            "category": "section",
                            "heading": section[0][3] if is_heading else "",
                            "header_level": header_level,
                            # A JSON string, because some vector stores only
                            # accept plain str/int/float metadata values.
                            "regions": json.dumps(
                                [
                                    {
                                        "page": page_index,
                                        "category": box["boxclass"],
                                        "bbox": _rounded_bbox(box),
                                    }
                                    for page_index, _, box, _ in section
                                ]
                            ),
                        }
                    ),
                )
            )
        return documents

    def _get_md(
        self,
        doc: pymupdf.Document,
        #pages: list | range | None = None,
    ) -> str:
        """Get the content of the page in markdown using PyMuPDF

        Args:
            doc: The PyMuPDF document object.
            pages: The pages. If omitted (None) all pages are processed.

        Returns:
            str: The content of the page in markdown.
        """
        try:
            import pymupdf 
        except ImportError:
            raise ImportError(
                "pymupdf package not found, please install it "
                "with `pip install pymupdf`"
            )

        pymupdf_params: dict[str, Any] = {
            **self.pymupdf_layout_kwargs,
        }

        md = doc.to_markdown(
            **pymupdf_params,
        )
        
        return md

    def _extract_metadata(self, doc: pymupdf.Document, blob: Blob) -> dict:
        """Extract metadata from the PDF document.

        Args:
            doc: The PyMuPDF document object.
            blob: The blob being parsed.

        Returns:
            dict: The extracted metadata from the PDF.
        """
        metadata = _purge_metadata(
            {
                **{
                    "producer": "PyMuPDF",
                    "creator": "PyMuPDF",
                    "creationdate": "",
                    "source": blob.source,
                    "file_path": blob.source,
                    "total_pages": len(doc),
                },
                **{
                    k: doc.metadata[k]
                    for k in doc.metadata
                    if isinstance(doc.metadata[k], (str, int))
                },
            }
        )
        for k in ("modDate", "creationDate"):
            if k in doc.metadata:
                metadata[k] = doc.metadata[k]
        return metadata
