import pytesseract

# Path to the local Tesseract executable (Windows)
pytesseract.pytesseract.tesseract_cmd = r"paste the path of tesseract.exe"

from pathlib import Path
import fitz  # PyMuPDF (PDF reading + rendering pages to images)
import docx  # python-docx (DOCX reading)
from PIL import Image  # Pillow (image handling)

# If you're on Windows and Tesseract isn't detected automatically, use this alternative path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Supported image extensions for OCR
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _read_txt(path: Path) -> str:
    """Read plain text file (.txt) safely and return cleaned text."""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _read_docx(path: Path) -> str:
    """Read a DOCX file and return all paragraph text as one string."""
    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs)
    return text.strip()


def _pdf_text_layer(doc: fitz.Document) -> str:
    """
    Extract selectable text from a PDF (text layer).
    If the PDF is scanned, this will usually return very little or empty text.
    """
    parts = []
    for page in doc:
        parts.append((page.get_text("text") or "").strip())
    return "\n".join([p for p in parts if p]).strip()


def _pdf_looks_scanned(doc: fitz.Document, text_min_chars: int = 80) -> bool:
    """
    Heuristic to guess whether a PDF is scanned (image-based) or digital (text-based).

    Logic:
    - If total extracted text is very small AND
    - Many pages contain images
    Then it's likely a scanned PDF.
    """
    total_text = 0
    pages_with_images = 0

    for page in doc:
        txt = (page.get_text("text") or "").strip()
        total_text += len(txt)

        imgs = page.get_images(full=True)
        if len(imgs) > 0:
            pages_with_images += 1

    if doc.page_count == 0:
        return False

    # Consider it image-heavy if at least half the pages have images
    majority_images = pages_with_images >= max(1, doc.page_count // 2)

    # If text is tiny and PDF is image-heavy, treat as scanned
    return (total_text < text_min_chars) and majority_images


def _ocr_pil_image(img: Image.Image, lang: str = "eng", psm: int = 6) -> str:
    """
    Run Tesseract OCR on a PIL image.

    psm=6 means: assume a single uniform block of text.
    """
    config = f"--psm {psm}"
    return (pytesseract.image_to_string(img, lang=lang, config=config) or "").strip()


def _ocr_pdf(doc: fitz.Document, lang: str = "eng", dpi: int = 300, psm: int = 6) -> str:
    """
    OCR a PDF by rendering each page into an image and passing it to Tesseract.

    Steps:
    - Render each page at the requested DPI (higher DPI = better OCR but slower)
    - Convert rendered page into a PIL image
    - OCR each page and concatenate results
    """
    parts = []
    zoom = dpi / 72  # PDF default rendering base is 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = _ocr_pil_image(img, lang=lang, psm=psm)
        if text:
            parts.append(text)

    return "\n\n".join(parts).strip()


def extract_text_auto(file_path: str, ocr_lang: str = "eng", ocr_dpi: int = 300) -> dict:
    """
    Automatically extract text from different file types:
    - .txt  -> direct read
    - .docx -> parse paragraphs
    - .pdf  -> try text layer first, then OCR if scanned
    - images -> OCR

    Returns:
      {
        "text": "...",
        "method": "txt|docx|pdf_text|pdf_ocr|image_ocr|unsupported",
        "is_scanned_pdf": bool
      }

    ocr_lang examples:
      - "eng"
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Plain text
    if ext == ".txt":
        return {"text": _read_txt(path), "method": "txt", "is_scanned_pdf": False}

    # DOCX
    if ext == ".docx":
        return {"text": _read_docx(path), "method": "docx", "is_scanned_pdf": False}

    # PDF
    if ext == ".pdf":
        doc = fitz.open(str(path))
        try:
            text = _pdf_text_layer(doc)
            scanned = _pdf_looks_scanned(doc)

            # If we got enough text and it doesn't look scanned, use the text layer
            if text and not scanned:
                return {"text": text, "method": "pdf_text", "is_scanned_pdf": False}

            # Otherwise, OCR the PDF
            ocr_text = _ocr_pdf(doc, lang=ocr_lang, dpi=ocr_dpi, psm=6)
            return {"text": ocr_text, "method": "pdf_ocr", "is_scanned_pdf": True}
        finally:
            doc.close()

    # Image OCR
    if ext in SUPPORTED_IMAGE_EXTS:
        img = Image.open(str(path)).convert("RGB")
        text = _ocr_pil_image(img, lang=ocr_lang, psm=6)
        return {"text": text, "method": "image_ocr", "is_scanned_pdf": False}

    # Unsupported file type
    return {"text": "", "method": "unsupported", "is_scanned_pdf": False}


if __name__ == "__main__":
    # Example usage
    r = extract_text_auto(
        r"C:\Users\MCC\Desktop\Resume Screening\data\cvs\3c891da3-1138-eb11-9240-00155d404d09_bb2724b9c537432288b32ffbb7bc2be3.pdf",
        ocr_lang="eng",
        ocr_dpi=300
    )

    # Print extraction method and whether it was considered scanned
    print(r["method"], r["is_scanned_pdf"])

    # Print extracted text
    print(r["text"])
