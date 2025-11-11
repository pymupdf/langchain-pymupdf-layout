# langchain-pymupdf-layout
An integration package connecting PyMuPDF Layout to LangChain.

Load PDF content to Markdown using AI-based, CPU only, layout analysis.

## Installation

```bash
# Install in editable mode for development
pip install -e .

# Or build and install
pip install build
python -m build
pip install dist/langchain_pymupdf_layout-0.1.0-py3-none-any.whl
```

## Usage

```python
from langchain_pymupdf_layout import version

print(version())  # Output: version number
```

## Run the simple test

Uses the `PyMuPDFLayoutLoader` to load a smple PDF and prints the results.

```python
python simple_test.py
```

## Development

1. Clone or extract this package
2. Install in editable mode: `pip install -e .`
3. Make your changes
4. Build: `python -m build`

## Customize

- Edit `pyproject.toml` to change package name, version, dependencies
- Add your code to `langchain_pymupdf_layout/` directory
- Add more modules as needed
