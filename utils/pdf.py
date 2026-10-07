import pdfplumber


def extract_text(pdf_bytes: bytes) -> str:
    import io
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No extractable text found. PDF may be a scanned image.")
    return text
