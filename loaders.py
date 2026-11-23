from pathlib import Path

from langchain_core.documents import Document
from pypdf import PasswordType, PdfReader
from pypdf.errors import PyPdfError


class LoadError(Exception):
    """A problem with the input file, described in one friendly sentence."""


def load_txt(path):
    """One Document for the whole text file."""
    text = Path(path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": Path(path).name})]


def load_pdf(path):
    """One Document per page that has text."""
    reader = PdfReader(path)
    if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
        raise LoadError("This PDF is locked with a password.")

    docs = []
    for number, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": Path(path).name, "page": number},
                )
            )
    return docs


def load_document(path):
    """Load a .txt or .pdf file. Raises LoadError with a clear message on any problem."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".txt", ".pdf"):
        raise LoadError("Only .txt and .pdf files are supported.")

    try:
        docs = load_txt(path) if suffix == ".txt" else load_pdf(path)
    except FileNotFoundError:
        raise LoadError(
            "File not found. Please check that the path is correct.")
    except UnicodeDecodeError:
        raise LoadError(
            "The file is not UTF-8. Open it in Notepad, choose 'Save As', and set Encoding to UTF-8.")
    except PyPdfError:
        raise LoadError("The PDF cannot be read. The file may be damaged.")
    except OSError:
        raise LoadError(
            "The file cannot be opened. Check whether another program has it open.")

    if not any(doc.page_content.strip() for doc in docs):
        raise LoadError(
            "No text was found in the file. (A scanned PDF needs OCR.)")
    return docs
